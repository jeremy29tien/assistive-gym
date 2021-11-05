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
parser.add_argument('--models_dir', default = ".", help="top directory where checkpoint models for demos are stored")
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
reward_net.load_state_dict(torch.load(reward_net_path))
reward_net.to(device)

# print out predicted cumulative returns and actual returns
with torch.no_grad():
    pred_returns = [predict_traj_return(reward_net, traj) for traj in sorted_demos]
    best_pred_reward_sequence = predict_reward_sequence(reward_net, sorted_demos[-1])
    worst_pred_reward_sequence = predict_reward_sequence(reward_net, sorted_demos[0])

# for i, p in enumerate(pred_returns):
#     print(i, p, sorted_demo_rewards[i])

# print(best_pred_reward_sequence)
# print(sorted_demo_rewards_per_timestep[-1])

best_pred_cum_reward_sequence = []
cum_sum = 0
for el in best_pred_reward_sequence:
    cum_sum += el
    best_pred_cum_reward_sequence.append(cum_sum)
best_actual_cum_reward_sequence = []
cum_sum = 0
for el in sorted_demo_rewards_per_timestep[-1]:
    cum_sum += el
    best_actual_cum_reward_sequence.append(cum_sum)

worst_pred_cum_reward_sequence = []
cum_sum = 0
for el in worst_pred_reward_sequence:
    cum_sum += el
    worst_pred_cum_reward_sequence.append(cum_sum)
worst_actual_cum_reward_sequence = []
cum_sum = 0
for el in sorted_demo_rewards_per_timestep[0]:
    cum_sum += el
    worst_actual_cum_reward_sequence.append(cum_sum)


plot1 = plt.figure(1)
plt.plot(best_pred_reward_sequence, label='pred')
plt.plot(sorted_demo_rewards_per_timestep[-1], label='actual')
plt.xlabel("Timestep")
plt.ylabel("Reward")
plt.legend()
plt.savefig("reward_sequence_best.png")

plot2 = plt.figure(2)
plt.plot(best_pred_cum_reward_sequence, label='pred')
plt.plot(best_actual_cum_reward_sequence, label='actual')
plt.xlabel("Timestep")
plt.ylabel("Cumulative Reward")
plt.legend()
plt.savefig("cum_reward_sequence_best.png")

plot3 = plt.figure(3)
plt.plot(worst_pred_reward_sequence, label='pred')
plt.plot(sorted_demo_rewards_per_timestep[0], label='actual')
plt.xlabel("Timestep")
plt.ylabel("Reward")
plt.legend()
plt.savefig("reward_sequence_worst.png")

plot4 = plt.figure(4)
plt.plot(worst_pred_cum_reward_sequence, label='pred')
plt.plot(worst_actual_cum_reward_sequence, label='actual')
plt.xlabel("Timestep")
plt.ylabel("Cumulative Reward")
plt.legend()
plt.savefig("cum_reward_sequence_worst.png")

plot5 = plt.figure(5)
plt.plot(pred_returns, label='pred_returns')
plt.plot(sorted_demo_rewards, label='actual_returns')
plt.xlabel("Demo ranking")
plt.ylabel("Reward")
plt.legend()
plt.savefig("pred-v-actual_rewards.png")

plt.show()
