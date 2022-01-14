import assistive_gym
import gym
import pybullet as p
import numpy as np
import csv
import importlib
import multiprocessing, ray
from assistive_gym.learn import load_policy

FILE_NAME = "data/500demos_pretrained_augmentedfeatures.csv"
ENV_NAME = "FeedingSawyer-v1"
COOP = False
ADD_LINEAR_FEATURES = True

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

# Set up file for appending
file = open(FILE_NAME, 'a+', newline='')
write = csv.writer(file)

num_demos = 500
for demo in range(num_demos):
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
            # Compute the next action using the trained policy
            action = test_agent.compute_action(observation)

            # Collect the data
            # print("Observation:", observation)
            # print("Action:", action)

            if ADD_LINEAR_FEATURES:
                # Handtuned features: spoon-mouth distance, amount of food particles in mouth, amount of food particles on the floor
                distance = np.linalg.norm(observation[7:10])
                if info is None:
                    foods_in_mouth = 0
                    foods_on_floor = 0
                else:
                    foods_in_mouth = info['foods_in_mouth']
                    foods_on_floor = info['foods_on_ground']
                linear_data = np.array([distance, foods_in_mouth, foods_on_floor])
                data = np.concatenate((observation, linear_data, action))
            else:
                # Raw observations (+ actions)
                data = np.concatenate((observation, action))

            # Step the simulation forward using the action from our trained policy
            observation, reward, done, info = env.step(action)

        # Write data to file
        # print("Data:", data.tolist())
        write.writerow(data.tolist())

        # print("Reward:", reward)
        # print("Task Success:", info['task_success'])
        # print("\n")

env.disconnect()
file.close()
print("data appended to", FILE_NAME)
