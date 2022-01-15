import gym, assistive_gym
import pybullet as p
import numpy as np
import csv
import torch
from model import MLP
from model import predict
import time
import argparse

# NOTE: Settings for standard data: "feedingsawyer_standard.csv_model", True, 25
# NOTE: Settings for handmade data: "feedingsawyer_handmade.csv_model", False, 14

parser = argparse.ArgumentParser(description=None)
parser.add_argument('--model_path', default='',
                    help="name and location for learned model params, e.g. ./learned_models/breakout.params")
parser.add_argument('--seed', default=3, help="random seed for experiments")
parser.add_argument('--num_rollouts', default=100, type=int, help="number of rollouts")
parser.add_argument('--augmented', dest='augmented', default=False, action='store_true', help="whether we are using augmented features")  # NOTE: type=bool doesn't work, value is still true.
parser.add_argument('--render', dest='render', default=False, action='store_true', help="whether to render rollouts")  # NOTE: type=bool doesn't work, value is still true.
args = parser.parse_args()

## PARAMS ##
model_path = args.model_path
seed = args.seed
num_rollouts = args.num_rollouts
augmented = args.augmented
render = args.render
#################


# Set the random seed for reproducibility
torch.manual_seed(seed)
env = gym.make('FeedingSawyer-v1')
env.seed(seed)

# Augmented denotes a (observation, linear_features, action) dataset. Input dim would be 28 if true.
if augmented:
    input_dim = 28
else:
    input_dim = 25

# Whether to render rollouts
if render:
    env.render()

STANDARD_DATA_MODE = True  # ie., (observation, action). False denotes the handmade data.
VERBOSE = False

# Load model
model = MLP(input_dim)
model.load_state_dict(torch.load(model_path))

# reward_success[i, 0] contains the reward for the ith rollout
# reward_success[i, 1] contains the 1 if success, 0 if fail for the ith rollout
reward_success = np.zeros((num_rollouts, 2))

# Define success as completing the task within timeout == 30 seconds
# Note that timeout can be adjusted depending on whether or not we are rendering the graphics
# EDIT: we can actually use the `done` variable to set a bound on when to call it quits
# timeout = 10


for i in range(num_rollouts):
    observation = env.reset()
    total_reward = 0

    # timeout_start = time.time()
    # while time.time() < timeout_start + timeout:
    info = None
    done = False
    while not done:
        if STANDARD_DATA_MODE:
            if augmented:
                # Handtuned features: spoon-mouth distance, amount of food particles in mouth, amount of food particles on the floor
                distance = np.linalg.norm(observation[7:10])
                if info is None:
                    foods_in_mouth = 0
                    foods_on_floor = 0
                else:
                    foods_in_mouth = info['foods_in_mouth']
                    foods_on_floor = info['foods_on_ground']
                linear_data = np.array([distance, foods_in_mouth, foods_on_floor])
                input = np.concatenate((observation, linear_data))
            else:
                input = observation
        else:
            robot_state = np.concatenate((env.robot.get_pos_orient(env.robot.right_end_effector)[0],
                                          env.robot.get_pos_orient(env.robot.right_end_effector)[1]))
            human_state = np.concatenate((env.human.get_pos_orient(env.human.head)[0],
                                          env.human.get_pos_orient(env.human.head)[1]))
            input = np.concatenate((robot_state, human_state))

        pred_action = predict(input, model)[0]
        if VERBOSE:
            print('input:', input)
            print('pred_action:', pred_action)

        # Step the simulation forward
        observation, reward, done, info = env.step(pred_action)
        total_reward += reward
        if VERBOSE:
            print("Reward:", reward)
            print("Total Reward:", total_reward)
            print("Task Success:", info['task_success'])
            print("\n")
        if info['task_success']:
            reward_success[i, 1] = 1
            break
    # NOTE: If the task was successful, reward_success[i, 1] = 1 before exiting the loop.
    # Otherwise, reward_success[i, 1] = 0 (as initialized). Thus, we just have to set the reward.
    reward_success[i, 0] = total_reward
    print("Rollout %d Reward: %f" % (i, total_reward))

print(reward_success)
max_reward = np.amax(reward_success[:,0])
min_reward = np.amin(reward_success[:,0])
avg_reward = np.mean(reward_success[:,0])
std_err_reward = np.std(reward_success[:,0]) / np.sqrt(num_rollouts)
success_rate = np.sum(reward_success[:,1])/num_rollouts
print("max_reward:", max_reward)
print("min_reward:", min_reward)
print("avg_reward:", avg_reward)
print("std_err_reward:", std_err_reward)
print("success_rate:", success_rate)


