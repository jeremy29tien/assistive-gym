import assistive_gym
import gym
import pybullet as p
import numpy as np
import csv
import importlib
import multiprocessing, ray
from matplotlib import pyplot as plt
from assistive_gym.learn import load_policy
import argparse

# NOTE: Most of this is shamelessly copied from render_policy in learn.py.
# Link: https://github.com/Healthcare-Robotics/assistive-gym/blob/fb799c377e1f144ff96044fb9096725f7f9cfc61/assistive_gym/learn.py#L96


def make_env(env_name, seed=1001):
    env = gym.make('assistive_gym:'+env_name)
    env.seed(seed)
    return env


def generate_rollout_data(policy_path, data_dir, seed, num_rollouts, noisy, augmented, fully_observable, state_action, render):
    ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False)

    # Set up the environment
    env = make_env(ENV_NAME, seed=seed)  # fixed seed for reproducibility (1000 for training, 1001 for testing)

    # Load pretrained policy from file
    algo = 'ppo'

    test_agent, _ = load_policy(env, algo, ENV_NAME, policy_path, coop=False, seed=seed)

    if render:
        env.render()

    if noisy:
        noise_levels = [0, 0.2, 0.4, 0.6, 0.8, 1]
    else:
        noise_levels = [0]

    demos = []  # collection of trajectories
    total_rewards = []  # final reward at the end of a trajectory/demo
    rewards_over_time = []  # rewards at each timestep for each trajectory
    cum_rewards_over_time = []  # cumulative reward at each timestep for each trajectory, with a separate noise dimension
    rewards_per_noise_level = []  # final reward at the end of each trajectory, with a separate noise dimension
    for i, noise_level in enumerate(noise_levels):
        cum_rewards_over_time.append([])
        rewards_per_noise_level.append([])

        num_demos = num_rollouts
        for demo in range(num_demos):
            traj = []
            cum_reward_over_time = []
            reward_over_time = []
            total_reward = 0
            observation = env.reset()
            info = None
            done = False
            while not done:
                # Take random action with probability noise_level
                if np.random.rand() < noise_level:
                    action = env.action_space.sample()
                else:
                    # Compute the next action using the trained policy
                    action = test_agent.compute_action(observation)

                # Collect the data
                # print("Observation:", observation)
                # print("Action:", action)

                # FeedingSawyer
                # augmented (privileged) features: spoon-mouth distance, amount of food particles in mouth, amount of food particles on the floor
                # fully-observable: add previous end effector position, robot force on human, food information
                if ENV_NAME == "FeedingSawyer-v1":
                    distance = np.linalg.norm(observation[7:10])
                    if info is None:
                        foods_in_mouth = 0
                        foods_on_floor = 0
                    else:
                        foods_in_mouth = info['foods_in_mouth']
                        foods_on_floor = info['foods_on_ground']
                    handpicked_features = np.array([distance, foods_in_mouth, foods_on_floor])

                # ScratchItchJaco privileged features: end effector - target distance, total force at target
                if ENV_NAME == "ScratchItchJaco-v1":
                    distance = np.linalg.norm(observation[7:10])
                    if info is None:
                        tool_force_at_target = 0.0
                    else:
                        tool_force_at_target = info['tool_force_at_target']
                    handpicked_features = np.array([distance, tool_force_at_target])

                if augmented and state_action:
                    data = np.concatenate((observation, action, handpicked_features))
                elif augmented:
                    data = np.concatenate((observation, handpicked_features))
                elif state_action:
                    data = np.concatenate((observation, action))
                else:
                    data = observation

                # Step the simulation forward using the action from our trained policy
                observation, reward, done, info = env.step(action)

                traj.append(data)
                total_reward += reward
                reward_over_time.append(reward)
                cum_reward_over_time.append(total_reward)
                # print("Reward:", reward)
                # print("Task Success:", info['task_success'])
                # print("\n")
            demos.append(traj)

            cum_rewards_over_time[i].append(cum_reward_over_time)
            rewards_per_noise_level[i].append(total_reward)

            # print(total_reward)
            total_rewards.append(total_reward)
            rewards_over_time.append(reward_over_time)


    env.disconnect()
    rewards_per_noise_level = np.array(rewards_per_noise_level)
    mean_rewards_per_noise_level = np.mean(rewards_per_noise_level, axis=1)

    demos = np.asarray(demos)
    total_rewards = np.asarray(total_rewards)
    rewards_over_time = np.asarray(rewards_over_time)
    # print(demos)
    # print(total_rewards)

    if data_dir != '':
        np.save(data_dir+"/demos.npy", demos)
        np.save(data_dir+"/demo_rewards.npy", total_rewards)
        np.save(data_dir+"/demo_reward_per_timestep.npy", rewards_over_time)


    with np.printoptions(precision=3):
        print(rewards_per_noise_level)
        print(mean_rewards_per_noise_level)

    # Code for plotting how cumulative reward changes over time with each level of noise.
    # plt.figure()
    #
    # plt.subplot(231)
    # for p in cum_rewards_over_time[0]:
    #     plt.plot(p, 'b')
    #
    # plt.subplot(232)
    # for p in cum_rewards_over_time[1]:
    #     plt.plot(p, 'b')
    #
    # plt.subplot(233)
    # for p in cum_rewards_over_time[2]:
    #     plt.plot(p, 'b')
    #
    # plt.subplot(234)
    # for p in cum_rewards_over_time[3]:
    #     plt.plot(p, 'b')
    #
    # plt.subplot(235)
    # for p in cum_rewards_over_time[4]:
    #     plt.plot(p, 'b')
    #
    # plt.subplot(236)
    # for p in cum_rewards_over_time[5]:
    #     plt.plot(p, 'b')
    #
    # plt.savefig("cum_rewards_over_time.png")
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--data_dir', default='', help="location for generated rollouts")
    parser.add_argument('--policy_path', default='', help="location for (pre)trained policy")
    parser.add_argument('--seed', default=0, type=int, help="random seed for experiments")
    parser.add_argument('--num_rollouts', default=20, type=int, help="number of rollouts")
    parser.add_argument('--noisy', dest='noisy', default=False, action='store_true', help="whether we add noise to rollouts")
    parser.add_argument('--state_action', dest='state_action', default=False, action='store_true', help="whether data consists of state-action pairs rather that just states")  # NOTE: type=bool doesn't work, value is still true.
    parser.add_argument('--augmented', dest='augmented', default=False, action='store_true', help="whether data consists of states + linear features pairs rather that just states")  # NOTE: type=bool doesn't work, value is still true.
    parser.add_argument('--render', dest='render', default=False, action='store_true', help="whether to render rollouts")  # NOTE: type=bool doesn't work, value is still true.
    args = parser.parse_args()

    env = args.env
    data_dir = args.data_dir
    policy_path = args.policy_path
    seed = args.seed
    num_rollouts = args.num_rollouts
    noisy = args.noisy
    state_action = args.state_action
    augmented = args.augmented
    render = args.render

    if env == "feeding":
        ENV_NAME = "FeedingSawyer-v1"
    elif env == "scratching":
        ENV_NAME = "ScratchItchJaco-v1"

    generate_rollout_data(policy_path, data_dir, seed, num_rollouts, noisy, augmented, state_action, render)
