import pickle
import gym
import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


def create_training_data(demonstrations, num_trajs, num_snippets, min_snippet_length, max_snippet_length):
    # collect training data
    max_traj_length = 0
    training_obs = []
    training_labels = []
    num_demos = len(demonstrations)

    # add full trajs (for use on Enduro)
    for n in range(num_trajs):
        ti = 0
        tj = 0
        # only add trajectories that are different returns
        while (ti == tj):
            # pick two random demonstrations
            ti = np.random.randint(num_demos)
            tj = np.random.randint(num_demos)
        # create random partial trajs by finding random start frame and random skip frame
        si = np.random.randint(6)
        sj = np.random.randint(6)
        step = np.random.randint(3, 7)

        traj_i = demonstrations[ti][si::step]  # slice(start,stop,step)
        traj_j = demonstrations[tj][sj::step]

        if ti > tj:
            label = 0
        else:
            label = 1

        training_obs.append((traj_i, traj_j))
        training_labels.append(label)
        max_traj_length = max(max_traj_length, len(traj_i), len(traj_j))

    # fixed size snippets with progress prior
    for n in range(num_snippets):
        ti = 0
        tj = 0
        # only add trajectories that are different returns
        while (ti == tj):
            # pick two random demonstrations
            ti = np.random.randint(num_demos)
            tj = np.random.randint(num_demos)
        # create random snippets
        # find min length of both demos to ensure we can pick a demo no earlier than that chosen in worse preferred demo
        min_length = min(len(demonstrations[ti]), len(demonstrations[tj]))
        rand_length = np.random.randint(min_snippet_length, max_snippet_length)
        if ti < tj:  # pick tj snippet to be later than ti
            ti_start = np.random.randint(min_length - rand_length + 1)
            # print(ti_start, len(demonstrations[tj]))
            tj_start = np.random.randint(ti_start, len(demonstrations[tj]) - rand_length + 1)
        else:  # ti is better so pick later snippet in ti
            tj_start = np.random.randint(min_length - rand_length + 1)
            # print(tj_start, len(demonstrations[ti]))
            ti_start = np.random.randint(tj_start, len(demonstrations[ti]) - rand_length + 1)
        traj_i = demonstrations[ti][ti_start:ti_start + rand_length:2]  # skip everyother framestack to reduce size
        traj_j = demonstrations[tj][tj_start:tj_start + rand_length:2]

        max_traj_length = max(max_traj_length, len(traj_i), len(traj_j))
        if ti > tj:
            label = 0
        else:
            label = 1
        training_obs.append((traj_i, traj_j))
        training_labels.append(label)

    print("maximum traj length", max_traj_length)
    return training_obs, training_labels


observation_dim = 25

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(observation_dim, 128)
        self.fc2 = nn.Linear(128, 64)  # Added a hidden layer for additional expressiveness
        self.fc3 = nn.Linear(64, 1)


    def cum_return(self, traj):
        '''calculate cumulative return of trajectory'''
        sum_rewards = 0
        #compute forward pass of reward network (we parallelize across frames so batch size is length of full trajectory)
        x = F.leaky_relu(self.fc1(traj))
        x = F.leaky_relu(self.fc2(x))
        r = self.fc3(x)
        sum_rewards += torch.sum(r)
        return sum_rewards



    def forward(self, traj_i, traj_j):
        '''compute cumulative return for each trajectory and return logits'''
        cum_r_i = self.cum_return(traj_i)
        cum_r_j = self.cum_return(traj_j)
        return torch.cat((cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)),0)


def learn_reward(reward_network, optimizer, training_inputs, training_outputs, num_iter, l1_reg, checkpoint_dir):
    # check if gpu available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print(device)
    loss_criterion = nn.CrossEntropyLoss()

    cum_loss = 0.0
    training_data = list(zip(training_inputs, training_outputs))
    for epoch in range(num_iter):
        np.random.shuffle(training_data)
        training_obs, training_labels = zip(*training_data)
        for i in range(len(training_labels)):
            traj_i, traj_j = training_obs[i]
            labels = np.array([training_labels[i]])
            traj_i = np.array(traj_i)
            traj_j = np.array(traj_j)
            traj_i = torch.from_numpy(traj_i).float().to(device)
            traj_j = torch.from_numpy(traj_j).float().to(device)
            labels = torch.from_numpy(labels).to(device)

            # zero out gradient
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, abs_rewards = reward_network.forward(traj_i, traj_j)
            outputs = outputs.unsqueeze(0)
            loss = loss_criterion(outputs, labels) + l1_reg * abs_rewards
            loss.backward()
            optimizer.step()

            # print stats to see if learning
            item_loss = loss.item()
            cum_loss += item_loss
            if i % 100 == 99:
                # print(i)
                print("epoch {}:{} loss {}".format(epoch, i, cum_loss))
                print(abs_rewards)
                cum_loss = 0.0
                print("check pointing")
                torch.save(reward_net.state_dict(), checkpoint_dir)
    print("finished training")


def calc_accuracy(reward_network, training_inputs, training_outputs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_criterion = nn.CrossEntropyLoss()
    num_correct = 0.
    with torch.no_grad():
        for i in range(len(training_inputs)):
            label = training_outputs[i]
            traj_i, traj_j = training_inputs[i]
            traj_i = np.array(traj_i)
            traj_j = np.array(traj_j)
            traj_i = torch.from_numpy(traj_i).float().to(device)
            traj_j = torch.from_numpy(traj_j).float().to(device)

            #forward to get logits
            outputs, abs_return = reward_network.forward(traj_i, traj_j)
            _, pred_label = torch.max(outputs,0)
            if pred_label.item() == label:
                num_correct += 1.
    return num_correct / len(training_inputs)