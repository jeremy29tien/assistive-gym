import assistive_gym
import gym
import pybullet as p
import numpy as np
import csv
import importlib
import multiprocessing, ray
from matplotlib import pyplot as plt
from assistive_gym.learn import load_policy

ENV_NAME = "FeedingSawyer-v1"
COOP = False

# NOTE: Most of this is shamelessly copied from render_policy in learn.py.
# Link: https://github.com/Healthcare-Robotics/assistive-gym/blob/fb799c377e1f144ff96044fb9096725f7f9cfc61/assistive_gym/learn.py#L96


def make_env(env_name, coop=False, seed=1001):
    if not coop:
        env = gym.make('assistive_gym:'+env_name)
    else:
        module = importlib.import_module('assistive_gym.envs')
        env_class = getattr(module, env_name.split('-')[0] + 'Env')
        env = env_class()
    env.seed(seed)
    return env

ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False)

# Set up the environment
env = make_env(ENV_NAME, coop=COOP, seed=1000)  # fixed seed for reproducibility (1000 for training, 1001 for testing)

# Load pretrained policy from file
algo = 'ppo'

# /Users/jeremytien/Documents/3rd-Year/Research/Anca Dragan/assistive-gym/trained_models/ppo/FeedingSawyer-v1/checkpoint_521/checkpoint-521 for local
# /home/jtien/assistive-gym/trained_models/ppo/FeedingSawyer-v1/checkpoint_521/checkpoint-521 for server
policy_path = '/home/jtien/assistive-gym/trained_models/ppo/FeedingSawyer-v1/checkpoint_521/checkpoint-521'
test_agent, _ = load_policy(env, algo, ENV_NAME, policy_path, COOP, seed=1000)

# env.render()

noise_levels = [0, 0.2, 0.4, 0.6, 0.8, 1]
demos = []  # collection of trajectories
total_rewards = []  # final reward at the end of a trajectory/demo
rewards_over_time = []  # rewards at each timestep for each trajectory
cum_rewards_over_time = []  # cumulative reward at each timestep for each trajectory, with a separate noise dimension
rewards_per_noise_level = []  # final reward at the end of each trajectory, with a separate noise dimension
for i, noise_level in enumerate(noise_levels):
    cum_rewards_over_time.append([])
    rewards_per_noise_level.append([])

    num_demos = 20
    for demo in range(num_demos):
        traj = []
        cum_reward_over_time = []
        reward_over_time = []
        total_reward = 0
        observation = env.reset()
        info = None
        done = False
        while not done:
            if COOP:
                # Compute the next action for the robot/human using the trained policies
                action_robot = test_agent.compute_action(observation['robot'], policy_id='robot')
                action_human = test_agent.compute_action(observation['human'], policy_id='human')

                # Collect the data
                data = np.concatenate((observation['robot'], observation['human'], action_robot, action_human))

                # Step the simulation forward using the actions from our trained policies
                observation, reward, done, info = env.step({'robot': action_robot, 'human': action_human})
                done = done['__all__']
            else:
                # Take random action with probability noise_level
                if np.random.rand() < noise_level:
                    action = env.action_space.sample()
                else:
                    # Compute the next action using the trained policy
                    action = test_agent.compute_action(observation)

                # Collect the data
                # print("Observation:", observation)
                # print("Action:", action)

                # Raw state: observation + action
                # data = np.concatenate((observation, action))

                # Handtuned features: spoon-mouth distance, amount of food particles in mouth, amount of food particles on the floor
                distance = np.linalg.norm(observation[7:10])
                if info is None:
                    foods_in_mouth = 0
                    foods_on_floor = 0
                else:
                    foods_in_mouth = info['foods_in_mouth']
                    foods_on_floor = info['foods_on_ground']
                data = np.array([distance, foods_in_mouth, foods_on_floor])

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

np.save("data/handpicked_features/demos.npy", demos)
np.save("data/handpicked_features/demo_rewards.npy", total_rewards)
np.save("data/handpicked_features/demo_reward_per_timestep.npy", rewards_over_time)


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

