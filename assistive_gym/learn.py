import os, sys, multiprocessing, gym, ray, shutil, argparse, importlib, glob
import numpy as np
# from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.rllib.agents import ppo, sac
from ray.rllib.utils.typing import PolicyID
from ray.tune.logger import pretty_print
from numpngw import write_apng
from tensorboardX import SummaryWriter
from ray.rllib.agents.callbacks import DefaultCallbacks
from typing import Dict, Optional
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.agents.callbacks import DefaultCallbacks


# https://stackoverflow.com/questions/41117740/tensorflow-crashes-with-cublas-status-alloc-failed
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


class CustomCallbacks(DefaultCallbacks):

    def on_episode_start(self,
                         *,
                         worker: "RolloutWorker",
                         base_env: BaseEnv,
                         policies: Dict[PolicyID, Policy],
                         episode: MultiAgentEpisode,
                         env_index: Optional[int] = None,
                         **kwargs) -> None:
        print("episode {} started".format(episode.episode_id))
        episode.user_data['gt_rewards'] = []
        episode.hist_data['gt_rewards'] = []

    def on_episode_step(self,
                        *,
                        worker: "RolloutWorker",
                        base_env: BaseEnv,
                        episode: MultiAgentEpisode,
                        env_index: Optional[int] = None,
                        **kwargs) -> None:
        info = episode.last_info_for()
        if info and 'gt_reward' in info:
            r = info['gt_reward']
            episode.user_data['gt_rewards'].append(r)

    def on_episode_end(self,
                       *,
                       worker: "RolloutWorker",
                       base_env: BaseEnv,
                       policies: Dict[PolicyID, Policy],
                       episode: MultiAgentEpisode,
                       env_index: Optional[int] = None,
                       **kwargs) -> None:
        gt_reward = np.sum(episode.user_data["gt_rewards"])
        print("episode {} ended with length {} and gt reward {}".format(
            episode.episode_id, episode.length, gt_reward))
        episode.custom_metrics["gt_reward"] = gt_reward
        episode.hist_data["gt_rewards"] = episode.user_data["gt_rewards"]

    # def on_train_result(self, *, trainer, result: dict, **kwargs) -> None:
    #     result['gt_reward'] = None


def setup_config(env, algo, coop=False, seed=0, extra_configs={}):
    num_processes = multiprocessing.cpu_count()
    if algo == 'ppo':
        config = ppo.DEFAULT_CONFIG.copy()
        config['train_batch_size'] = 19200
        config['num_sgd_iter'] = 50
        config['sgd_minibatch_size'] = 128
        config['lambda'] = 0.95
        config['model']['fcnet_hiddens'] = [100, 100]
    elif algo == 'sac':
        # NOTE: pip3 install tensorflow_probability
        config = sac.DEFAULT_CONFIG.copy()
        config['timesteps_per_iteration'] = 400
        config['learning_starts'] = 1000
        config['Q_model']['fcnet_hiddens'] = [100, 100]
        config['policy_model']['fcnet_hiddens'] = [100, 100]
        # config['normalize_actions'] = False
    config['num_workers'] = num_processes
    print('num_workers:', num_processes)
    config['num_cpus_per_worker'] = 0
    config['seed'] = seed
    config['log_level'] = 'ERROR'
    config['callbacks'] = CustomCallbacks
    # if algo == 'sac':
    #     config['num_workers'] = 1
    if coop:
        obs = env.reset()
        policies = {'robot': (None, env.observation_space_robot, env.action_space_robot, {}), 'human': (None, env.observation_space_human, env.action_space_human, {})}
        config['multiagent'] = {'policies': policies, 'policy_mapping_fn': lambda a: a}
        config['env_config'] = {'num_agents': 2}
    return {**config, **extra_configs}


def load_policy(env, algo, env_name, policy_path=None, coop=False, seed=0, extra_configs={}):
    if algo == 'ppo':
        agent = ppo.PPOTrainer(setup_config(env, algo, coop, seed, extra_configs), 'assistive_gym:'+env_name)  # 'assistive_gym:'+env_name
    elif algo == 'sac':
        agent = sac.SACTrainer(setup_config(env, algo, coop, seed, extra_configs), 'assistive_gym:'+env_name)  # 'assistive_gym:'+env_name
    if policy_path != '':
        if 'checkpoint' in policy_path:
            agent.restore(policy_path)
            print("##################")
            print("Loading directly from a specific policy path:", policy_path)
            print("##################")
        else:
            # Find the most recent policy in the directory
            directory = os.path.join(policy_path, algo, env_name)
            files = [f.split('_')[-1] for f in glob.glob(os.path.join(directory, 'checkpoint_*'))]
            files_ints = [int(f) for f in files]
            if files:
                checkpoint_max = max(files_ints)
                checkpoint_num = files_ints.index(checkpoint_max)
                checkpoint_path = os.path.join(directory, 'checkpoint_%s' % files[checkpoint_num], 'checkpoint-%d' % checkpoint_max)
                agent.restore(checkpoint_path)
                print("##################")
                print("Inferring policy to load based on env_name:", checkpoint_path)
                print("##################")

                # return agent, checkpoint_path
            return agent, None
    return agent, None

def make_env(env_name, coop=False, seed=1001, reward_net_path=None, indvar=None):
    if not coop and reward_net_path is not None and indvar is not None:
        env = gym.make('assistive_gym:'+env_name, reward_net_path=reward_net_path, indvar=indvar)
    elif not coop and reward_net_path is not None:
        env = gym.make('assistive_gym:' + env_name, reward_net_path=reward_net_path)
    elif not coop:
        env = gym.make('assistive_gym:'+env_name)
    else:
        module = importlib.import_module('assistive_gym.envs')
        env_class = getattr(module, env_name.split('-')[0] + 'Env')
        env = env_class()
    env.seed(seed)
    return env

def train(env_name, algo, evalonly_env_name='', timesteps_total=1000000, save_dir='./trained_models/', load_policy_path='', coop=False, seed=0, save_checkpoints=False, reward_net_path=None, evalonly_reward_net_path=None, indvar=None, extra_configs={}, tb=False):
    ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False)
    env = make_env(env_name, coop, seed=seed, reward_net_path=reward_net_path, indvar=indvar)
    if reward_net_path is not None and indvar is not None:
        agent, checkpoint_path = load_policy(env, algo, env_name, load_policy_path, coop, seed, extra_configs={"env_config": {"reward_net_path": reward_net_path, "indvar": indvar}})
    elif reward_net_path is not None:
        agent, checkpoint_path = load_policy(env, algo, env_name, load_policy_path, coop, seed, extra_configs={
            "env_config": {"reward_net_path": reward_net_path}})
    else:
        agent, checkpoint_path = load_policy(env, algo, env_name, load_policy_path, coop, seed)

    env.disconnect()

    timesteps = 0
    while timesteps < timesteps_total:
        result = agent.train()
        timesteps = result['timesteps_total']
        if coop:
            # Rewards are added in multi agent envs, so we divide by 2 since agents share the same reward in coop
            result['episode_reward_mean'] /= 2
            result['episode_reward_min'] /= 2
            result['episode_reward_max'] /= 2
        print(f"Iteration: {result['training_iteration']}, total timesteps: {result['timesteps_total']}, total time: {result['time_total_s']:.1f}, FPS: {result['timesteps_total']/result['time_total_s']:.1f}, mean reward: {result['episode_reward_mean']:.1f}, min/max reward: {result['episode_reward_min']:.1f}/{result['episode_reward_max']:.1f}")
        print("Custom metrics:", result['custom_metrics'])
        sys.stdout.flush()
        if tb:
            writer.add_scalar('scalar/' + env_name + '_reward', result['episode_reward_mean'], timesteps)
            writer.add_scalar('scalar/' + env_name + '_GTreward', result['custom_metrics']['gt_reward_mean'], timesteps)

        if not (save_checkpoints and result['training_iteration'] % 10 == 1):
            # Delete the old saved policy
            if checkpoint_path is not None:
                shutil.rmtree(os.path.dirname(checkpoint_path), ignore_errors=True)

        # Save the recently trained policy
        checkpoint_path = agent.save(os.path.join(save_dir, algo, env_name))
        # if tb:
        #     aux_reward, _ = evaluate_policy(evalonly_env_name, algo, checkpoint_path, n_episodes=1, seed=seed, verbose=False, reward_net_path=evalonly_reward_net_path)
        #     writer.add_scalar('scalar/'+evalonly_env_name+'_reward', aux_reward, timesteps)

    return checkpoint_path

def render_policy(env, env_name, algo, policy_path, coop=False, colab=False, seed=0, n_episodes=1, extra_configs={}):
    ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False)
    if env is None:
        env = make_env(env_name, coop, seed=seed)
        if colab:
            env.setup_camera(camera_eye=[0.5, -0.75, 1.5], camera_target=[-0.2, 0, 0.75], fov=60, camera_width=1920//4, camera_height=1080//4)
    print(policy_path)
    test_agent, _ = load_policy(env, algo, env_name, policy_path, coop, seed, extra_configs)

    if not colab:
        env.render()
    frames = []
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        reward_total = 0.0
        while not done:
            if coop:
                # Compute the next action for the robot/human using the trained policies
                action_robot = test_agent.compute_action(obs['robot'], policy_id='robot')
                action_human = test_agent.compute_action(obs['human'], policy_id='human')
                # Step the simulation forward using the actions from our trained policies
                obs, reward, done, info = env.step({'robot': action_robot, 'human': action_human})
                done = done['__all__']
            else:
                # Compute the next action using the trained policy
                action = test_agent.compute_action(obs)
                # Step the simulation forward using the action from our trained policy
                obs, reward, done, info = env.step(action)
                reward_total += reward
            if colab:
                # Capture (render) an image from the camera
                img, depth = env.get_camera_image_depth()
                frames.append(img)
        print('Reward total: %.2f' % (reward_total))
    env.disconnect()
    if colab:
        filename = 'output_%s.png' % env_name
        write_apng(filename, frames, delay=100)
        return filename

def evaluate_policy(env_name, algo, policy_path, n_episodes=100, coop=False, seed=0, verbose=False, reward_net_path=None, indvar=None, extra_configs={}):
    ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False)
    env = make_env(env_name, coop, seed=seed, reward_net_path=reward_net_path, indvar=indvar)
    if reward_net_path is not None and indvar is not None:
        test_agent, _ = load_policy(env, algo, env_name, policy_path, coop, seed, extra_configs={"env_config": {"reward_net_path": reward_net_path, "indvar": indvar}})
    elif reward_net_path is not None:
        test_agent, _ = load_policy(env, algo, env_name, policy_path, coop, seed, extra_configs={
            "env_config": {"reward_net_path": reward_net_path}})
    else:
        test_agent, _ = load_policy(env, algo, env_name, policy_path, coop, seed, extra_configs)

    rewards = []
    forces = []
    task_successes = []
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        reward_total = 0.0
        force_list = []
        task_success = 0.0
        while not done:
            if coop:
                # Compute the next action for the robot/human using the trained policies
                action_robot = test_agent.compute_action(obs['robot'], policy_id='robot')
                action_human = test_agent.compute_action(obs['human'], policy_id='human')
                # Step the simulation forward using the actions from our trained policies
                obs, reward, done, info = env.step({'robot': action_robot, 'human': action_human})
                reward = reward['robot']
                done = done['__all__']
                info = info['robot']
            else:
                action = test_agent.compute_action(obs)
                obs, reward, done, info = env.step(action)
            reward_total += reward
            force_list.append(info['total_force_on_human'])
            task_success = info['task_success']

        rewards.append(reward_total)
        forces.append(np.mean(force_list))
        task_successes.append(task_success)
        if verbose:
            print('Reward total: %.2f, mean force: %.2f, task success: %r' % (reward_total, np.mean(force_list), task_success))
        sys.stdout.flush()
    env.disconnect()

    print('\n', '-'*50, '\n')
    # print('Rewards:', rewards)
    print('Reward Mean:', np.mean(rewards))
    print('Reward Std:', np.std(rewards))

    # print('Forces:', forces)
    print('Force Mean:', np.mean(forces))
    print('Force Std:', np.std(forces))

    # print('Task Successes:', task_successes)
    print('Task Success Mean:', np.mean(task_successes))
    print('Task Success Std:', np.std(task_successes))
    sys.stdout.flush()

    return np.mean(rewards), np.std(rewards), np.mean(task_successes), np.std(task_successes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL for Assistive Gym')
    parser.add_argument('--env', default='ScratchItchJaco-v0',
                        help='Environment to train on (default: ScratchItchJaco-v0)')
    parser.add_argument('--algo', default='ppo',
                        help='Reinforcement learning algorithm')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed (default: 1)')
    parser.add_argument('--train', action='store_true', default=False,
                        help='Whether to train a new policy')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Whether to render a single rollout of a trained policy')
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='Whether to evaluate a trained policy over n_episodes')
    parser.add_argument('--train-timesteps', type=int, default=1000000,
                        help='Number of simulation timesteps to train a policy (default: 1000000)')
    parser.add_argument('--save-dir', default='./trained_models/',
                        help='Directory to save trained policy in (default ./trained_models/)')
    parser.add_argument('--load-policy-path', default='./trained_models/',
                        help='Path name to saved policy checkpoint (NOTE: Use this to continue training an existing policy, or to evaluate a trained policy)')
    parser.add_argument('--render-episodes', type=int, default=1,
                        help='Number of rendering episodes (default: 1)')
    parser.add_argument('--eval-episodes', type=int, default=100,
                        help='Number of evaluation episodes (default: 100)')
    parser.add_argument('--colab', action='store_true', default=False,
                        help='Whether rendering should generate an animated png rather than open a window (e.g. when using Google Colab)')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Whether to output more verbose prints')
    parser.add_argument('--save-checkpoints', action='store_true', default=False,
                        help='Whether to save multiple checkpoints of trained policy')
    parser.add_argument('--reward-net-path', default=None,
                        help='Path name to trained reward network.')
    # parser.add_argument('--indvar', type=int, default=-1, nargs='+',
    #                     help='Placeholder to pass in independent variable for experiments.')
    parser.add_argument('--indvar', default=None, nargs='+',
                        help='Placeholder to pass in independent variable for experiments.')
    parser.add_argument('--tb', default=False, action='store_true', help='Use tensorboardX?')
    args = parser.parse_args()

    if args.tb:
        writer = SummaryWriter()

    coop = ('Human' in args.env)
    checkpoint_path = None

    if args.train:
        checkpoint_path = train(args.env, args.algo, timesteps_total=args.train_timesteps, save_dir=args.save_dir, load_policy_path=args.load_policy_path, coop=coop, seed=args.seed, save_checkpoints=args.save_checkpoints, reward_net_path=args.reward_net_path, indvar=tuple(args.indvar) if args.indvar is not None else None, tb=args.tb)
    if args.render:
        render_policy(None, args.env, args.algo, checkpoint_path if checkpoint_path is not None else args.load_policy_path, coop=coop, colab=args.colab, seed=args.seed, n_episodes=args.render_episodes)
    if args.evaluate:
        evaluate_policy(args.env, args.algo, checkpoint_path if checkpoint_path is not None else args.load_policy_path, n_episodes=args.eval_episodes, coop=coop, seed=args.seed, verbose=args.verbose, reward_net_path=args.reward_net_path, indvar=tuple(args.indvar) if args.indvar is not None else None)

    if args.tb:
        writer.close()
