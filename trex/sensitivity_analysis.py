import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from sklearn.model_selection import train_test_split
from trex.model import Net, predict_traj_return, predict_reward_sequence
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=None)
parser.add_argument('--reward_net_path', default='', help="name and location for learned model params")
parser.add_argument('--seed', default=0, help="random seed for experiments")
parser.add_argument('--save_fig_dir', help ="where to save visualizations")

args = parser.parse_args()

reward_net_path = args.reward_net_path

demos = np.load("data/demos.npy")
demo_rewards = np.load("data/demo_rewards.npy")
demo_reward_per_timestep = np.load("data/demo_reward_per_timestep.npy")

# sorts the demos in order of increasing reward (most negative reward to most positive reward)
# note that sorted_demos is now a python list, not a np array
sorted_demos = [x for _, x in sorted(zip(demo_rewards, demos), key=lambda pair: pair[0])]

sorted_demo_rewards_per_timestep = [x for _, x in sorted(zip(demo_rewards, demo_reward_per_timestep), key=lambda pair: pair[0])]

sorted_demo_rewards = sorted(demo_rewards)

# Now we create a reward network and optimize it using the training data.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
reward_net = Net()
reward_net.load_state_dict(torch.load(reward_net_path, map_location='cpu'))
reward_net.to(device)

with torch.no_grad():
    # Find the 2 best states and the 2 worst states (in terms of our predicted reward)
    max_reward = 0
    second_max_reward = 0
    min_reward = 1000
    second_min_reward = 1000

    best_state = None
    second_best_state = None
    worst_state = None
    second_worst_state = None
    for traj in sorted_demos:
        reward_seq = predict_reward_sequence(reward_net, traj)
        if max(reward_seq) > max_reward:
            max_reward = max(reward_seq)
            best_state = traj[np.argmax(reward_seq)]
        if min(reward_seq) < min_reward:
            min_reward = min(reward_seq)
            worst_state = traj[np.argmin(reward_seq)]
    for traj in sorted_demos:
        reward_seq = predict_reward_sequence(reward_net, traj)
        if max(reward_seq) > second_max_reward and max(reward_seq) != max_reward:
            second_max_reward = max(reward_seq)
            second_best_state = traj[np.argmax(reward_seq)]
        if min(reward_seq) < second_min_reward and min(reward_seq) != min_reward:
            second_min_reward = min(reward_seq)
            second_worst_state = traj[np.argmin(reward_seq)]

with np.printoptions(precision=4, suppress=True):
    print("max_reward", max_reward)
    print("best_state", best_state)
    print()
    print("second_max_reward", second_max_reward)
    print("second_best_state", second_best_state)
    print()
    print("second_min_reward", second_min_reward)
    print("second_worst_state", second_worst_state)
    print()
    print("min_reward", min_reward)
    print("worst_state", worst_state)
    print()


def sensitivity(name, state, reward):
    most_sens_feature = None
    max_dev_from_orig_reward = 0
    print("Sensitivity analysis on the " + name + " state...")
    for i, feature in enumerate(state):
        print("Feature", i)
        state[i] = 5
        print("set to 5:")
        print("reward:", predict_reward_sequence(reward_net, [state]))
        print("difference from original:", predict_reward_sequence(reward_net, [state])[0] - reward)

        state[i] = 0
        print("set to 0:")
        print("reward:", predict_reward_sequence(reward_net, [state]))
        delta = abs(predict_reward_sequence(reward_net, [state])[0] - reward)
        print("difference from original:", delta)
        if delta > max_dev_from_orig_reward:
            max_dev_from_orig_reward = delta
            most_sens_feature = i

        state[i] = -5
        print("set to -5:")
        print("reward:", predict_reward_sequence(reward_net, [state]))
        print("difference from original:", predict_reward_sequence(reward_net, [state])[0] - reward)
        print()

        state[i] = feature
    return most_sens_feature


s1 = sensitivity("best", best_state, max_reward)
s2 = sensitivity("second best", second_best_state, second_max_reward)
s3 = sensitivity("second worst", second_worst_state, second_min_reward)
s4 = sensitivity("worst", worst_state, min_reward)

print("Looking at sensitivity to zeroing out features...")
print("Most sensitive feature for best_state:", s1)
print("Most sensitive feature for second_best_state:", s2)
print("Most sensitive feature for second_worst_state:", s3)
print("Most sensitive feature for worst_state:", s4)