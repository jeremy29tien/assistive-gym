import gym, assistive_gym
import pybullet as p
import numpy as np
import csv
import torch
from model import MLP
from model import predict
import time

# NOTE: Settings for standard data: "feedingsawyer_standard.csv_model", True, 25
# NOTE: Settings for handmade data: "feedingsawyer_handmade.csv_model", False, 14

FILE_NAME = "models/feedingsawyer_standard4.csv.model"
STANDARD_DATA_MODE = True
input_dim = 25
VERBOSE = False

env = gym.make('FeedingSawyer-v1')
env.set_seed(1001)  # fixed seed for reproducibility (1000 for training, 1001 for testing)
# env.render()

# Load model
model = MLP(input_dim)
model.load_state_dict(torch.load(FILE_NAME))

num_rollouts = 100
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
    done = False
    while not done:
        if STANDARD_DATA_MODE:
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


