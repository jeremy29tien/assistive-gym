import pickle
import gym
import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from sklearn.model_selection import train_test_split


# num_comps specifies the number of trajectories to use in our training set
# pair_delta=1 recovers original (just that pairwise comps can't be the same)
# if all_pairs=True, rather than generating num_comps pairwise comps with pair_delta ranking difference,
# we simply generate all (num_demos choose 2) possible pairs from the dataset.
def create_training_data(demonstrations, num_comps, pair_delta, all_pairs):
    # collect training data
    max_traj_length = 0
    training_obs = []
    training_labels = []
    num_demos = len(demonstrations)

    if all_pairs:
        for ti in range(num_demos):
            for tj in range(ti+1, num_demos):
                traj_i = demonstrations[ti]
                traj_j = demonstrations[tj]

                # In other words, label = (traj_i < traj_j)
                if ti > tj:
                    label = 0  # 0 indicates that traj_i is better than traj_j
                else:
                    label = 1  # 1 indicates that traj_j is better than traj_i

                training_obs.append((traj_i, traj_j))
                training_labels.append(label)

                # We shouldn't need max_traj_length, since all our trajectories our fixed at length 200.
                max_traj_length = max(max_traj_length, len(traj_i), len(traj_j))

    else:
        for n in range(num_comps):
            ti = 0
            tj = 0
            # only add trajectories that are different (in sorted reward ranking) by pair_delta
            while abs(ti - tj) < pair_delta:
                # pick two random demonstrations
                ti = np.random.randint(num_demos)
                tj = np.random.randint(num_demos)

            traj_i = demonstrations[ti]
            traj_j = demonstrations[tj]

            # In other words, label = (traj_i < traj_j)
            if ti > tj:
                label = 0  # 0 indicates that traj_i is better than traj_j
            else:
                label = 1  # 1 indicates that traj_j is better than traj_i

            training_obs.append((traj_i, traj_j))
            training_labels.append(label)

            # We shouldn't need max_traj_length, since all our trajectories our fixed at length 200.
            max_traj_length = max(max_traj_length, len(traj_i), len(traj_j))

    print("maximum traj length", max_traj_length)
    return training_obs, training_labels


# NOTE: the 'handpicked' features are comprised of
# 1) spoon-mouth distance
# 2) amount of food particles in mouth
# 3) amount of food particles on the floor
class Net(nn.Module):
    def __init__(self, augmented=False, state_action=False):
        super().__init__()

        if augmented and state_action:
            input_dim = 35
        elif augmented:
            input_dim = 28
        elif state_action:
            input_dim = 32
        else:
            input_dim = 3

        self.fc1 = nn.Linear(input_dim, 1, bias=False)  # We have a single linear layer, with no nonlinearities


    def cum_return(self, traj):
        '''calculate cumulative return of trajectory'''
        sum_rewards = 0
        #compute forward pass of reward network (we parallelize across frames so batch size is length of full trajectory)
        r = self.fc1(traj)
        sum_rewards += torch.sum(r)
        return sum_rewards



    def forward(self, traj_i, traj_j):
        '''compute cumulative return for each trajectory and return logits'''
        cum_r_i = self.cum_return(traj_i)
        cum_r_j = self.cum_return(traj_j)
        return torch.cat((cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)),0)


def learn_reward(reward_network, optimizer, training_inputs, training_outputs, num_iter, l1_reg, checkpoint_dir, val_obs, val_labels, patience):
    # check if gpu available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print(device)
    loss_criterion = nn.CrossEntropyLoss()

    cum_loss = 0.0
    trigger_times = 0
    prev_min_val_loss = 100
    training_data = list(zip(training_inputs, training_outputs))
    for epoch in range(num_iter):
        np.random.shuffle(training_data)
        training_obs, training_labels = zip(*training_data)
        for i in range(len(training_labels)):
            traj_i, traj_j = training_obs[i]

            # Question: why was it called labels?
            label = np.array([training_labels[i]])
            traj_i = np.array(traj_i)
            traj_j = np.array(traj_j)
            traj_i = torch.from_numpy(traj_i).float().to(device)
            traj_j = torch.from_numpy(traj_j).float().to(device)
            label = torch.from_numpy(label).to(device)

            # zero out gradient
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = reward_network.forward(traj_i, traj_j)
            outputs = outputs.unsqueeze(0)
            # print("train outputs", outputs.shape)
            # print("train label", label.shape)
            loss = loss_criterion(outputs, label)  # got rid of the l1_reg * abs_rewards from this line
            loss.backward()
            optimizer.step()


            # print stats to see if learning
            item_loss = loss.item()
            cum_loss += item_loss
            if i % 100 == 99:
                val_loss = calc_val_loss(reward_network, val_obs, val_labels)
                val_acc = calc_accuracy(reward_network, val_obs, val_labels)
                print("epoch {}:{} loss {}, val_loss {}, val_acc {}".format(epoch, i, cum_loss, val_loss, val_acc))
                cum_loss = 0.0
                print("check pointing")
                print("Weights:", reward_net.state_dict())
                torch.save(reward_net.state_dict(), checkpoint_dir)

        val_loss = calc_val_loss(reward_network, val_obs, val_labels)
        val_acc = calc_accuracy(reward_network, val_obs, val_labels)
        print("end of epoch {}: val_loss {}, val_acc {}".format(epoch, val_loss, val_acc))
        torch.save(reward_net.state_dict(), checkpoint_dir)

        # Early Stopping
        if val_loss > prev_min_val_loss:
            trigger_times += 1
            print('trigger times:', trigger_times)
            if trigger_times >= patience:
                print("Early stopping.")
                print("Trained Weights:", reward_net.state_dict())
                return
        else:
            trigger_times = 0
            print('trigger times:', trigger_times)

        prev_min_val_loss = min(prev_min_val_loss, val_loss)
    print("finished training")
    print("Trained Weights:", reward_net.state_dict())


def calc_val_loss(reward_network, training_inputs, training_outputs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_criterion = nn.CrossEntropyLoss()
    losses = []
    with torch.no_grad():
        for i in range(len(training_inputs)):
            label = np.array([training_outputs[i]])
            traj_i, traj_j = training_inputs[i]
            traj_i = np.array(traj_i)
            traj_j = np.array(traj_j)
            traj_i = torch.from_numpy(traj_i).float().to(device)
            traj_j = torch.from_numpy(traj_j).float().to(device)
            label = torch.from_numpy(label).to(device)

            #forward to get logits
            outputs = reward_network.forward(traj_i, traj_j)
            outputs = outputs.unsqueeze(0)
            # print("val outputs", outputs.shape)
            # print("val label", label.shape)

            loss = loss_criterion(outputs, label)
            losses.append(loss.item())

    return np.mean(losses)


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
            outputs = reward_network.forward(traj_i, traj_j)
            _, pred_label = torch.max(outputs,0)
            if pred_label.item() == label:
                num_correct += 1.
    return num_correct / len(training_inputs)


def predict_reward_sequence(net, traj):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rewards_from_obs = []
    with torch.no_grad():
        for s in traj:
            r = net.cum_return(torch.from_numpy(np.array([s])).float().to(device)).item()
            rewards_from_obs.append(r)
    return rewards_from_obs


def predict_traj_return(net, traj):
    return sum(predict_reward_sequence(net, traj))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    # parser.add_argument('--env_name', default='', help='Select the environment name to run, i.e. pong')
    parser.add_argument('--reward_model_path', default='',
                        help="name and location for learned model params, e.g. ./learned_models/breakout.params")
    parser.add_argument('--seed', default=0, help="random seed for experiments")
    parser.add_argument('--num_comps', default=0, type=int, help="number of pairwise comparisons")
    parser.add_argument('--num_demos', default=120, type=int, help="the number of demos to sample pairwise comps from")
    parser.add_argument('--num_epochs', default=100, type=int, help="number of training epochs")
    parser.add_argument('--lr', default=0.00005, type=float, help="learning rate")
    parser.add_argument('--weight_decay', default=0.0, type=float, help="weight decay")
    parser.add_argument('--patience', default=100, type=int, help="number of iterations we wait before early stopping")
    parser.add_argument('--pair_delta', default=10, type=int, help="min difference between trajectory rankings in our dataset")
    parser.add_argument('--all_pairs', dest='all_pairs', default=False, action='store_true', help="whether we generate all pairs from the dataset (num_demos choose 2)")  # NOTE: type=bool doesn't work, value is still true.
    parser.add_argument('--state_action', dest='state_action', default=False, action='store_true', help="whether data consists of state-action pairs rather that just states")  # NOTE: type=bool doesn't work, value is still true.
    parser.add_argument('--augmented', dest='augmented', default=False, action='store_true', help="whether data consists of states + linear features pairs rather that just states")  # NOTE: type=bool doesn't work, value is still true.
    args = parser.parse_args()

    seed = args.seed
    torch.manual_seed(seed)

    ## HYPERPARAMS ##
    num_comps = args.num_comps  # the number of pairwise comparisons we draw
    num_demos = args.num_demos
    lr = args.lr
    weight_decay = args.weight_decay
    num_iter = args.num_epochs  # num times through training data
    patience = args.patience
    pair_delta = args.pair_delta
    all_pairs = args.all_pairs
    state_action = args.state_action
    augmented = args.augmented
    l1_reg = 0.0
    #################

    if augmented and state_action:
        demos = np.load("data/augmented_stateactions/demos.npy")
        demo_rewards = np.load("data/augmented_stateactions/demo_rewards.npy")
        demo_reward_per_timestep = np.load("data/augmented_stateactions/demo_reward_per_timestep.npy")
    elif augmented:
        demos = np.load("data/augmented_features/demos.npy")
        demo_rewards = np.load("data/augmented_features/demo_rewards.npy")
        demo_reward_per_timestep = np.load("data/augmented_features/demo_reward_per_timestep.npy")
    else:
        demos = np.load("data/handpicked_features/demos.npy")
        demo_rewards = np.load("data/handpicked_features/demo_rewards.npy")
        demo_reward_per_timestep = np.load("data/handpicked_features/demo_reward_per_timestep.npy")

    demo_lengths = 200  # fixed horizon of 200 timesteps in assistive gym
    print("demo lengths", demo_lengths)

    print("demos:", demos.shape)
    print("demo_rewards:", demo_rewards.shape)

    # sort the demonstrations according to ground truth reward to simulate ranked demos
    # sorts the demos in order of increasing reward (most negative reward to most positive reward)
    # note that sorted_demos is now a python list, not a np array
    sorted_demos = [x for _, x in sorted(zip(demo_rewards, demos), key=lambda pair: pair[0])]
    sorted_demos = np.array(sorted_demos)
    sorted_demo_rewards = sorted(demo_rewards)
    sorted_demo_rewards = np.array(sorted_demo_rewards)
    print(sorted_demo_rewards)

    # Subsample the demos according to num_demos
    # Source: https://stackoverflow.com/questions/50685409/select-n-evenly-spaced-out-elements-in-array-including-first-and-last
    idx = np.round(np.linspace(0, len(demos) - 1, num_demos)).astype(int)
    sorted_demos = sorted_demos[idx]
    sorted_demo_rewards = sorted_demo_rewards[idx]
    demo_reward_per_timestep = demo_reward_per_timestep[idx]  # Note: not used.

    train_val_split_seed = seed
    obs, labels = create_training_data(sorted_demos, num_comps, pair_delta, all_pairs)
    if len(obs) > 1:
        training_obs, val_obs, training_labels, val_labels = train_test_split(obs, labels, test_size=0.10, random_state=train_val_split_seed)
    else:
        print("WARNING: Since there is only one training point, the validation data is the same as the training data.")
        training_obs = val_obs = obs
        training_labels = val_labels = labels

    print("num training_obs", len(training_obs))
    print("num training_labels", len(training_labels))
    print("num val_obs", len(val_obs))
    print("num val_labels", len(val_labels))

    # Now we create a reward network and optimize it using the training data.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reward_net = Net()
    reward_net.to(device)
    import torch.optim as optim

    optimizer = optim.Adam(reward_net.parameters(), lr=lr, weight_decay=weight_decay)
    learn_reward(reward_net, optimizer, training_obs, training_labels, num_iter, l1_reg, args.reward_model_path, val_obs, val_labels, patience)
    # save reward network
    torch.save(reward_net.state_dict(), args.reward_model_path)

    # print out predicted cumulative returns and actual returns
    with torch.no_grad():
        pred_returns = [predict_traj_return(reward_net, traj) for traj in sorted_demos]
    for i, p in enumerate(pred_returns):
        print(i, p, sorted_demo_rewards[i])

    print("train accuracy:", calc_accuracy(reward_net, training_obs, training_labels))
    print("validation accuracy:", calc_accuracy(reward_net, val_obs, val_labels))