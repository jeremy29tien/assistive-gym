import trex.model
import assistive_gym.learn
import argparse
import numpy as np
import multiprocessing, ray
import re, string
import sys

EVAL_SEED = 3


def get_rollouts(env_name, num_rollouts, policy_path, seed, pure_fully_observable=False, fully_observable=False, state_action=False):
    ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False)
    # Set up the environment
    env = assistive_gym.learn.make_env("FeedingSawyer-v1", seed=seed)
    # Load pretrained policy from file
    test_agent, _ = assistive_gym.learn.load_policy(env, 'ppo', "FeedingSawyer-v1", policy_path, seed=seed)

    new_rollouts = []
    new_rollout_rewards = []
    for r in range(num_rollouts):
        traj = []
        reward_total = 0.0
        obs = env.reset()
        info = None
        done = False
        while not done:
            action = test_agent.compute_action(obs)

            # FeedingSawyer
            # augmented (privileged) features: spoon-mouth distance, amount of food particles in mouth, amount of food particles on the floor
            # fully-observable: add previous end effector position, robot force on human, food information
            if env_name == "feeding":
                distance = np.linalg.norm(obs[7:10])
                if info is None:
                    foods_in_mouth = 0
                    foods_on_floor = 0
                    foods_hit_human = 0
                    sum_food_mouth_velocities = 0
                    prev_spoon_pos_real = np.zeros(3)
                    robot_force_on_human = 0
                else:
                    foods_in_mouth = info['foods_in_mouth']
                    foods_on_floor = info['foods_on_ground']
                    foods_hit_human = info['foods_hit_human']
                    sum_food_mouth_velocities = info['sum_food_mouth_velocities']
                    prev_spoon_pos_real = info['prev_spoon_pos_real']
                    robot_force_on_human = info['robot_force_on_human']
                privileged_features = np.array([distance, foods_in_mouth, foods_on_floor])
                fo_features = np.concatenate(([foods_in_mouth, foods_on_floor, foods_hit_human,
                                               sum_food_mouth_velocities], prev_spoon_pos_real, [robot_force_on_human]))
                # Features from the raw observation that are causal:
                # spoon_pos_real - target_pos_real and self.spoon_force_on_human, respectively
                pure_obs = np.concatenate((obs[7:10], obs[24:25]))

            # ScratchItchJaco privileged features: end effector - target distance, total force at target
            if env_name == "scratch_itch":
                distance = np.linalg.norm(obs[7:10])
                if info is None:
                    tool_force_at_target = 0.0
                    prev_tool_pos_real = np.zeros(3)
                    robot_force_on_human = 0
                    prev_tool_force = 0
                else:
                    tool_force_at_target = info['tool_force_at_target']
                    prev_tool_pos_real = info['prev_tool_pos_real']
                    robot_force_on_human = info['robot_force_on_human']
                    prev_tool_force = info['prev_tool_force']
                privileged_features = np.array([distance, tool_force_at_target])
                fo_features = np.concatenate((prev_tool_pos_real, [robot_force_on_human, prev_tool_force]))
                # Features from the raw observation that are causal:
                # tool_pos_real, tool_pos_real - target_pos_real, and self.tool_force, respectively
                pure_obs = np.concatenate((obs[0:3], obs[7:10], obs[29:30]))

            if pure_fully_observable:
                data = np.concatenate((pure_obs, action, fo_features))
            elif fully_observable:
                data = np.concatenate((obs, action, fo_features))
            elif state_action:
                data = np.concatenate((obs, action))
            else:
                data = obs

            obs, reward, done, info = env.step(action)

            traj.append(data)
            reward_total += reward

        new_rollouts.append(traj)
        new_rollout_rewards.append(reward_total)

    new_rollouts = np.asarray(new_rollouts)
    new_rollout_rewards = np.asarray(new_rollout_rewards)
    return new_rollouts, new_rollout_rewards


def run_active_learning(env, num_al_iter, mixing_factor, union_rollouts, retrain, seed, nn):
    np.random.seed(seed)

    # Load demonstrations from file and initialize pool of demonstrations
    if nn:
        if env == "feeding":
            demos = np.load("trex/data/raw_data/demos.npy")  # Currently, we are running a partially observable experiment
            demo_rewards = np.load("trex/data/raw_data/demo_rewards.npy")
        elif env == "scratch_itch":
            demos = np.load("trex/data/scratchitch/pure_fully_observable/demos.npy")
            demo_rewards = np.load("trex/data/scratchitch/pure_fully_observable/demo_rewards.npy")
    else:
        # FIXME: No linear case for Feeding or Itch Scratching yet
        demos = np.load("trex/data/augmented_full/demos.npy")
        demo_rewards = np.load("trex/data/augmented_full/demo_rewards.npy")
    num_demos = demos.shape[0]

    if mixing_factor is not None:
        regex = re.compile('[%s]' % re.escape(string.punctuation))
        config = env + "/" + "active_learning/" + str(num_al_iter) + "aliter_" + regex.sub('', str(mixing_factor)) + "mix_"
    elif union_rollouts is not None:
        config = env + "/" + "active_learning/" + str(num_al_iter) + "aliter_" + str(union_rollouts) + "union_"
    if retrain:
        if nn:
            if env == "feeding":
                config = config + "retrain_partiallyobservable_324demos_allpairs_hdim256-256-256_100epochs_10patience_0001lr_001weightdecay_seed" + str(seed)
            elif env == "scratch_itch":
                config = config + "retrain_120demos_hdim128-64_purefullyobservable_allpairs_100epochs_10patience_000001lr_00001weightdecay_seed" + str(seed)
        else:
            # FIXME: not relevant
            config = config + "retrain_augmentedfull_linear_2000prefs_2deltareward_100epochs_10patience_001lr_001l1reg_seed" + str(
                seed)
    else:
        if nn:
            if env == "feeding":
                config = config + "partiallyobservable_324demos_allpairs_hdim256-256-256_100epochs_10patience_0001lr_001weightdecay_seed" + str(seed)
            elif env == "scratch_itch":
                config = config + "120demos_hdim128-64_purefullyobservable_allpairs_100epochs_10patience_000001lr_00001weightdecay_seed" + str(seed)
        else:
            # FIXME: not relevant
            config = config + "augmentedfull_linear_2000prefs_2deltareward_100epochs_10patience_001lr_001l1reg_seed" + str(
                seed)

    reward_model_path = "/home/jeremy/assistive-gym/trex/models/" + config + ".params"
    reward_output_path = "/home/jeremy/assistive-gym/trex/reward_learning_outputs/" + config + ".txt"

    policy_save_dir = "./trained_models_reward_learning/" + config
    policy_eval_dir = "/home/jeremy/assistive-gym/trex/rl/eval/" + config

    rewards = []
    successes = []
    weights = []
    # For num_al_iter active learning iterations:
    for i in range(num_al_iter):
        # # 1. Run reward learning
        # with open(reward_output_path, 'a') as sys.stdout:
        #     # Use the al_data argument to input our pool of changing demonstrations
        #     if nn:
        #         if env == "feeding":
        #             final_weights = trex.model.run(reward_model_path, feeding=True, scratch_itch=False, seed=seed,
        #                                            hidden_dims=(256, 256, 256), num_demos=324, all_pairs=True,
        #                                            num_epochs=100, patience=10, lr=0.001, weight_decay=0.01,
        #                                            state_action=True,
        #                                            al_data=(demos, demo_rewards), load_weights=(not retrain),
        #                                            return_weights=False)
        #         elif env == "scratch_itch":
        #             final_weights = trex.model.run(reward_model_path, feeding=False, scratch_itch=True, seed=seed,
        #                                            hidden_dims=(128, 64), num_demos=120, all_pairs=True,
        #                                            num_epochs=100, patience=10, lr=0.00001, weight_decay=0.0001, pure_fully_observable=True,
        #                                            al_data=(demos, demo_rewards), load_weights=(not retrain), return_weights=False)
        #     else:
        #         # FIXME: not relevant
        #         final_weights = trex.model.run(reward_model_path, seed=seed, num_comps=2000, delta_reward=2,
        #                                        num_epochs=100, patience=10, lr=0.01, l1_reg=0.01, augmented_full=True,
        #                                        al_data=(demos, demo_rewards), load_weights=(not retrain), return_weights=True)
        # sys.stdout = sys.__stdout__  # reset stdout
        # if not nn:
        #     weights.append(final_weights['fcs.0.weight'].cpu().detach().numpy())
        #
        # # 2. Run RL (using the learned reward)
        # if retrain:
        #     checkpoint_path = assistive_gym.learn.train("FeedingLearnedRewardSawyer-v0", "ppo",
        #                                              timesteps_total=1000000, save_dir=policy_save_dir + "/" + str(i+1),
        #                                              load_policy_path='', seed=seed,
        #                                              reward_net_path=reward_model_path)
        # else:
        #     checkpoint_path = assistive_gym.learn.train("FeedingLearnedRewardSawyer-v0", "ppo", timesteps_total=((i+1)*1000000), save_dir=policy_save_dir, load_policy_path=policy_save_dir, seed=seed, reward_net_path=reward_model_path)
        #
        # # 3. Load RL policy, generate rollouts (number depends on mixing factor), and rank according to GT reward
        # if mixing_factor is not None:
        #     print("using mixing factor of", mixing_factor, "...")
        #     num_new_rollouts = round(num_demos * mixing_factor)
        # elif union_rollouts is not None:
        #     print("unioning", union_rollouts, "rollouts...")
        #     num_new_rollouts = union_rollouts
        # new_rollouts, new_rollout_rewards = get_rollouts(env, num_new_rollouts, checkpoint_path, seed, state_action=True)
        #
        # # 4. Based on mixing factor, sample (without replacement) demonstrations from previous iteration accordingly
        # if mixing_factor is not None:
        #     num_old_trajs = round(num_demos * (1 - mixing_factor))
        #     old_traj_i = np.random.choice(num_demos, size=num_old_trajs, replace=False)
        #     old_trajs = demos[old_traj_i]
        #     old_traj_rewards = demo_rewards[old_traj_i]
        # elif union_rollouts is not None:
        #     old_trajs = demos
        #     old_traj_rewards = demo_rewards
        #
        # # Update our pool of demonstrations
        # demos = np.concatenate((old_trajs, new_rollouts), axis=0)
        # demo_rewards = np.concatenate((old_traj_rewards, new_rollout_rewards), axis=0)

        checkpoint_path = policy_save_dir + "/" + str(i+1) + "/ppo/FeedingLearnedRewardSawyer-v0/checkpoint_40/checkpoint-40"
        # 5. Evaluate (latest) trained policy
        eval_path = policy_eval_dir + "/" + str(i+1) + ".txt"
        with open(eval_path, 'w') as sys.stdout:
            mean_reward, std_reward, mean_success, std_success = assistive_gym.learn.evaluate_policy("FeedingSawyer-v1", "ppo", checkpoint_path, n_episodes=100, seed=EVAL_SEED,
                                             verbose=True)
        sys.stdout = sys.__stdout__  # reset stdout
        rewards.append([mean_reward, std_reward])
        successes.append([mean_success, std_success])

    # NOTE: rewards[i] denotes the ith iteration of active learning. rewards[i][0] gives the reward mean,
    # and rewards[i][1] the std dev.
    # weights[i] contains the (linear) reward function weights at the end of the ith iteration.
    rewards = np.asarray(rewards)
    successes = np.asarray(successes)
    np.save(policy_eval_dir + "/" + "rewards.npy", rewards)
    np.save(policy_eval_dir + "/" + "successes.npy", successes)
    if not nn:
        weights = np.asarray(weights)
        np.save(policy_eval_dir + "/" + "weights.npy", weights)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env', default='scratch_itch', help='')
    parser.add_argument('--seed', default=0, type=int, help="seed")
    parser.add_argument('--num_al_iter', default=0, type=int, help="number of active learning iterations (where 1 is equivalent to normal pref-based reward learning")
    parser.add_argument('--mix', default=None, type=float, help="hyperparameter for how much to mix in new rollouts, where 1 means the next iteration consists of ONLY new rollouts")
    parser.add_argument('--union', default=None, type=int, help="hyperparameter for the number of rollouts from the new policy")
    parser.add_argument('--retrain', dest='retrain', default=False, action='store_true', help="whether to retrain reward and policy from scratch in each active learning iteration")
    parser.add_argument('--nn', dest='nn', default=False, action='store_true', help="whether to use a neural net for reward fn")


    args = parser.parse_args()

    env = args.env
    seed = args.seed
    num_al_iter = args.num_al_iter
    mixing_factor = args.mix
    union_rollouts = args.union
    retrain = args.retrain
    nn = args.nn

    run_active_learning(env, num_al_iter, mixing_factor, union_rollouts, retrain, seed, nn)



