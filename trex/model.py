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
from os.path import exists
from gpu_utils import determine_default_torch_device


# num_comps specifies the number of pairwise comparisons between trajectories to use in our training set
# delta_rank=1 recovers original (just that pairwise comps can't be the same)
# if all_pairs=True, rather than generating num_comps pairwise comps with delta_rank ranking difference,
# we simply generate all (num_demos choose 2) possible pairs from the dataset.
# Note: demonstrations must be sorted by increasing reward.
def create_training_data(sorted_demonstrations, sorted_rewards, num_comps=0, delta_rank=1, delta_reward=0, all_pairs=False):
    # collect training data
    max_traj_length = 0
    training_obs = []
    training_labels = []
    num_demos = len(sorted_demonstrations)

    if all_pairs:
        for ti in range(num_demos):
            for tj in range(ti+1, num_demos):
                traj_i = sorted_demonstrations[ti]
                traj_j = sorted_demonstrations[tj]

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
        # add full trajs
        for n in range(num_comps):
            ti = 0
            tj = 0
            if delta_reward == 0:
                while abs(ti - tj) < delta_rank:
                    # pick two random demonstrations
                    ti = np.random.randint(num_demos)
                    tj = np.random.randint(num_demos)
            else:
                while abs(sorted_rewards[ti] - sorted_rewards[tj]) < delta_reward:
                    # pick two random demonstrations
                    ti = np.random.randint(num_demos)
                    tj = np.random.randint(num_demos)

            traj_i = sorted_demonstrations[ti]
            traj_j = sorted_demonstrations[tj]

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


# NOTE:
# If input is comprised of state-action pairs, input_dim = 32
# If input is comprised of states, input_dim = 25
# input_dim = 25
class Net(nn.Module):
    def __init__(self, env, hidden_dims=(128,64), augmented=False, fully_observable=False, pure_fully_observable=False, new_fully_observable=False, new_pure_fully_observable=False, num_rawfeatures=25, state_action=False, norm=False):
        super().__init__()

        if new_pure_fully_observable:
            if env == "feeding":
                raise Exception("NOT IMPLEMENTED.")
            elif env == "scratch_itch":
                input_dim = 20
        if new_fully_observable:
            if env == "feeding":
                raise Exception("NOT IMPLEMENTED.")
            elif env == "scratch_itch":
                input_dim = 43
        elif pure_fully_observable:
            if env == "feeding":
                input_dim = 19
            elif env == "scratch_itch":
                input_dim = 19
        elif fully_observable:
            if env == "feeding":
                input_dim = 40
            elif env == "scratch_itch":
                input_dim = 42
        elif augmented and state_action:
            # Feeding only
            input_dim = 35
        elif augmented:
            if env == "feeding":
                input_dim = num_rawfeatures + 3
            elif env == "scratch_itch":
                input_dim = num_rawfeatures + 2
        elif state_action:
            # Feeding only
            input_dim = 32
        else:
            if env == "feeding":
                input_dim = 25
            elif env == "scratch_itch":
                input_dim = 30

        self.normalize = norm
        if self.normalize:
            print("Normalizing input features...")
            self.layer_norm = nn.LayerNorm(input_dim)
        self.num_layers = len(hidden_dims) + 1

        self.fcs = nn.ModuleList([None for _ in range(self.num_layers)])
        if len(hidden_dims) == 0:
            self.fcs[0] = nn.Linear(input_dim, 1, bias=False)
        else:
            self.fcs[0] = nn.Linear(input_dim, hidden_dims[0])
            for l in range(len(hidden_dims)-1):
                self.fcs[l+1] = nn.Linear(hidden_dims[l], hidden_dims[l+1])
            self.fcs[len(hidden_dims)] = nn.Linear(hidden_dims[-1], 1, bias=False)

        print(self.fcs)

    def cum_return(self, traj):
        '''calculate cumulative return of trajectory'''
        sum_rewards = 0
        #compute forward pass of reward network (we parallelize across frames so batch size is length of full trajectory)
        x = traj

        # Normalize features
        if self.normalize:
            x = self.layer_norm(x)

        for l in range(self.num_layers - 1):
            x = F.leaky_relu(self.fcs[l](x))
        r = self.fcs[-1](x)

        # Sum across 'batch', which is really the time dimension of the trajectory
        sum_rewards += torch.sum(r)
        return sum_rewards

    def forward(self, traj_i, traj_j):
        '''compute cumulative return for each trajectory and return logits'''
        cum_r_i = self.cum_return(traj_i)
        cum_r_j = self.cum_return(traj_j)
        return torch.cat((cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)),0)


def learn_reward(device, reward_network, optimizer, training_inputs, training_outputs, num_epochs, l1_reg, checkpoint_dir, val_obs, val_labels, patience, return_weights=False):
    print("device:", device)
    # Note that a sigmoid is implicitly applied in the CrossEntropyLoss
    loss_criterion = nn.CrossEntropyLoss()

    trigger_times = 0
    prev_min_val_loss = 100
    training_data = list(zip(training_inputs, training_outputs))
    final_weights = None
    for epoch in range(num_epochs):
        np.random.shuffle(training_data)
        training_obs, training_labels = zip(*training_data)
        for i in range(len(training_labels)):
            traj_i, traj_j = training_obs[i]

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

            # Calculate loss
            cross_entropy_loss = loss_criterion(outputs, label)
            l1_loss = l1_reg * torch.linalg.vector_norm(torch.cat([param.view(-1) for param in reward_network.parameters()]), 1)
            loss = cross_entropy_loss + l1_loss

            # Backpropagate
            loss.backward()

            # Take one optimizer step
            optimizer.step()

        val_loss = calc_val_loss(device, reward_network, val_obs, val_labels)
        val_acc = calc_accuracy(device, reward_network, val_obs, val_labels)
        print("end of epoch {}: val_loss {}, val_acc {}".format(epoch, val_loss, val_acc))

        # Early Stopping
        if val_loss > prev_min_val_loss:
            trigger_times += 1
            print('trigger times:', trigger_times)
            if trigger_times >= patience:
                print("Early stopping.")
                if return_weights:
                    return final_weights
                return None
        else:
            trigger_times = 0
            print('trigger times:', trigger_times)
            print("saving model weights...")
            torch.save(reward_network.state_dict(), checkpoint_dir)
            if return_weights:
                print("Weights:", reward_network.state_dict())
                final_weights = reward_network.state_dict()

        prev_min_val_loss = min(prev_min_val_loss, val_loss)
    print("Finished training.")
    if return_weights:
        return final_weights
    return None


# Calculates the cross-entropy losses over the entire validation set and returns the MEAN.
def calc_val_loss(device, reward_network, training_inputs, training_outputs):
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


def calc_accuracy(device, reward_network, training_inputs, training_outputs):
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


def predict_reward_sequence(device, net, traj):
    rewards_from_obs = []
    with torch.no_grad():
        for s in traj:
            r = net.cum_return(torch.from_numpy(np.array([s])).float().to(device)).item()
            rewards_from_obs.append(r)
    return rewards_from_obs


def predict_traj_return(device, net, traj):
    return sum(predict_reward_sequence(device, net, traj))


def run(reward_model_path, seed, feeding=True, scratch_itch=False, num_comps=0, num_demos=120, hidden_dims=tuple(), lr=0.00005, weight_decay=0.0, l1_reg=0.0,
        num_epochs=100, patience=100, delta_rank=1, delta_reward=0, all_pairs=False, augmented=False, fully_observable=False,
        pure_fully_observable=False, new_pure_fully_observable=False, new_fully_observable=False, num_rawfeatures=11, state_action=False, normalize_features=False, teleop=False, test=False,
        al_data=tuple(), load_weights=False, return_weights=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if al_data:
        demos = al_data[0]
        demo_rewards = al_data[1]
    else:
        if new_pure_fully_observable:
            if feeding:
                raise Exception("NOT IMPLEMENTED.")
            elif scratch_itch:
                demos = np.load("data/scratchitch/new_pure_fully_observable/demos.npy")
                demo_rewards = np.load("data/scratchitch/new_pure_fully_observable/demo_rewards.npy")
        elif new_fully_observable:
            if feeding:
                raise Exception("NOT IMPLEMENTED.")
            elif scratch_itch:
                demos = np.load("data/scratchitch/new_fully_observable/demos.npy")
                demo_rewards = np.load("data/scratchitch/new_fully_observable/demo_rewards.npy")
        elif pure_fully_observable:
            if feeding:
                demos = np.load("data/feeding/pure_fully_observable/demos.npy")
                demo_rewards = np.load("data/feeding/pure_fully_observable/demo_rewards.npy")
            elif scratch_itch:
                demos = np.load("data/scratchitch/pure_fully_observable/demos.npy")
                demo_rewards = np.load("data/scratchitch/pure_fully_observable/demo_rewards.npy")
        elif fully_observable:
            if feeding:
                demos = np.load("data/feeding/fully_observable/demos.npy")
                demo_rewards = np.load("data/feeding/fully_observable/demo_rewards.npy")
                demo_reward_per_timestep = np.load("data/feeding/fully_observable/demo_reward_per_timestep.npy")
            elif scratch_itch:
                demos = np.load("data/scratchitch/fully_observable/demos.npy")
                demo_rewards = np.load("data/scratchitch/fully_observable/demo_rewards.npy")
                demo_reward_per_timestep = np.load("data/scratchitch/fully_observable/demo_reward_per_timestep.npy")
        elif augmented and state_action:
            demos = np.load("data/augmented_stateactions/demos.npy")
            demo_rewards = np.load("data/augmented_stateactions/demo_rewards.npy")
            demo_reward_per_timestep = np.load("data/augmented_stateactions/demo_reward_per_timestep.npy")
        elif augmented:
            if teleop:
                demos = np.load("data/teleop/augmented/demos20.npy")
                demo_rewards = np.load("data/teleop/augmented/demo_rewards20.npy")
            elif scratch_itch:
                demos = np.load("data/scratchitch/augmented/demos.npy")
                demo_rewards = np.load("data/scratchitch/augmented/demo_rewards.npy")
                demo_reward_per_timestep = np.load("data/scratchitch/augmented/demo_reward_per_timestep.npy")
            else:
                demos = np.load("data/augmented_features/demos.npy")
                demo_rewards = np.load("data/augmented_features/demo_rewards.npy")
                demo_reward_per_timestep = np.load("data/augmented_features/demo_reward_per_timestep.npy")
            # NOTE: this active learning code is outdated.##########
            # if active_learning:
            #     if seed == 0:
            #         al_demos = np.load("data/active_learning/1770comps_60pairdelta_100epochs_10patience_001lr_01weightdecay_seed0/demos.npy")
            #         al_demo_rewards = np.load("data/active_learning/1770comps_60pairdelta_100epochs_10patience_001lr_01weightdecay_seed0/demo_rewards.npy")
            #         al_demo_reward_per_timestep = np.load("data/active_learning/1770comps_60pairdelta_100epochs_10patience_001lr_01weightdecay_seed0/demo_reward_per_timestep.npy")
            #     if seed == 1:
            #         al_demos = np.load("data/active_learning/1770comps_60pairdelta_100epochs_10patience_001lr_01weightdecay_seed1/demos.npy")
            #         al_demo_rewards = np.load("data/active_learning/1770comps_60pairdelta_100epochs_10patience_001lr_01weightdecay_seed1/demo_rewards.npy")
            #         al_demo_reward_per_timestep = np.load("data/active_learning/1770comps_60pairdelta_100epochs_10patience_001lr_01weightdecay_seed1/demo_reward_per_timestep.npy")
            #     if seed == 2:
            #         al_demos = np.load("data/active_learning/1770comps_60pairdelta_100epochs_10patience_001lr_01weightdecay_seed2/demos.npy")
            #         al_demo_rewards = np.load("data/active_learning/1770comps_60pairdelta_100epochs_10patience_001lr_01weightdecay_seed2/demo_rewards.npy")
            #         al_demo_reward_per_timestep = np.load("data/active_learning/1770comps_60pairdelta_100epochs_10patience_001lr_01weightdecay_seed2/demo_reward_per_timestep.npy")
            #
            #     demos = np.concatenate((demos, al_demos), axis=0)
            #     demo_rewards = np.concatenate((demo_rewards, al_demo_rewards), axis=0)
            #     demo_reward_per_timestep = np.concatenate((demo_reward_per_timestep, al_demo_reward_per_timestep), axis=0)
            #######################################################

            raw_features = demos[:, :, 0:num_rawfeatures]  # how many raw features to keep in the observation
            if scratch_itch:
                handpicked_features = demos[:, :, 30:32]  # handpicked features are the last 2
            elif feeding:
                handpicked_features = demos[:, :, 25:28]  # handpicked features are the last 3
            else:
                raise Exception("Need to specify either --feeding or --scratch_itch.")
            demos = np.concatenate((raw_features, handpicked_features), axis=-1)  # assign the result back to demos
        else:
            if teleop:
                demos = np.load("data/teleop/raw_states/demos.npy")
                demo_rewards = np.load("data/teleop/raw_states/demo_rewards.npy")
            elif scratch_itch:
                demos = np.load("data/scratchitch/raw/demos.npy")
                demo_rewards = np.load("data/scratchitch/raw/demo_rewards.npy")
                demo_reward_per_timestep = np.load("data/scratchitch/raw/demo_reward_per_timestep.npy")
            elif feeding:
                if state_action:
                    demos = np.load("data/raw_data/demos_stateactions.npy")
                else:
                    demos = np.load("data/raw_data/demos_states.npy")
                demo_rewards = np.load("data/raw_data/demo_rewards.npy")
                demo_reward_per_timestep = np.load("data/raw_data/demo_reward_per_timestep.npy")
            else:
                raise Exception("Need to specify either --feeding or --scratch_itch.")

            if test:
                # Test Data for Vanilla Model
                test_demos = np.load("data/raw_data/test_data/demos.npy")
                test_demo_rewards = np.load("data/raw_data/test_data/demo_rewards.npy")

    print("demos:", demos.shape)
    print("demo_rewards:", demo_rewards.shape)

    # Create disjoint set of validation trajectories
    idx = np.random.permutation(np.arange(demos.shape[0]))
    shuffled_demos = demos[idx]
    shuffled_rewards = demo_rewards[idx]
    train_val_split_i = int(demos.shape[0] * 0.1)
    val_demos = shuffled_demos[0:train_val_split_i]
    val_rewards = shuffled_rewards[0:train_val_split_i]
    train_demos = shuffled_demos[train_val_split_i:]
    train_rewards = shuffled_rewards[train_val_split_i:]

    # sort the demonstrations according to ground truth reward to simulate ranked demos
    # sorts the demos in order of increasing reward (most negative reward to most positive reward)
    # note that sorted_demos is now a python list, not a np array
    sorted_train_demos = np.array([x for _, x in sorted(zip(train_rewards, train_demos), key=lambda pair: pair[0])])
    sorted_train_rewards = np.array(sorted(train_rewards))
    # print("sorted_train_rewards:", sorted_train_rewards)

    sorted_val_demos = np.array([x for _, x in sorted(zip(val_rewards, val_demos), key=lambda pair: pair[0])])
    sorted_val_rewards = np.array(sorted(val_rewards))
    # print("sorted_val_rewards:", sorted_val_rewards)

    if test:
        # Sort test data as well
        sorted_test_demos = [x for _, x in sorted(zip(test_demo_rewards, test_demos), key=lambda pair: pair[0])]
        sorted_test_demos = np.array(sorted_test_demos)
        sorted_test_demo_rewards = sorted(test_demo_rewards)
        sorted_test_demo_rewards = np.array(sorted_test_demo_rewards)

    # Subsample the demos according to num_demos
    # Source: https://stackoverflow.com/questions/50685409/select-n-evenly-spaced-out-elements-in-array-including-first-and-last
    idx = np.round(np.linspace(0, len(sorted_train_demos) - 1, num_demos)).astype(int)
    sorted_train_demos = sorted_train_demos[idx]
    sorted_train_rewards = sorted_train_rewards[idx]
    # demo_reward_per_timestep = demo_reward_per_timestep[idx]  # Note: not used.

    train_obs, train_labels = create_training_data(sorted_train_demos, sorted_train_rewards, num_comps=num_comps, delta_rank=delta_rank, delta_reward=delta_reward, all_pairs=all_pairs)
    val_obs, val_labels = create_training_data(sorted_val_demos, sorted_val_rewards, all_pairs=True)

    print("num train_obs", len(train_obs))
    print("num train_labels", len(train_labels))
    print("num val_obs", len(val_obs))
    print("num val_labels", len(val_labels))
    if test:
        print("num test_obs", len(test_obs))
        print("num test_labels", len(test_labels))

    # Now we create a reward network and optimize it using the training data.
    device = torch.device(determine_default_torch_device(not torch.cuda.is_available()))

    reward_net = Net("scratch_itch" if scratch_itch else "feeding", hidden_dims=hidden_dims, augmented=augmented,  new_pure_fully_observable=new_pure_fully_observable, new_fully_observable=new_fully_observable, pure_fully_observable=pure_fully_observable, fully_observable=fully_observable, num_rawfeatures=num_rawfeatures, state_action=state_action, norm=normalize_features)

    # Check if we already trained this model before. If so, load the saved weights.
    if load_weights:
        model_exists = exists(reward_model_path)
        if model_exists:
            print("Found existing model weights! Loading state dict...")
            reward_net.load_state_dict(torch.load(reward_model_path))  # map_location=torch.device('cpu') may be necessary
        else:
            print("Could not find existing model weights. Training from scratch...")
    else:
        print("Training reward model from scratch...")

    reward_net.to(device)
    num_total_params = sum(p.numel() for p in reward_net.parameters())
    num_trainable_params = sum(p.numel() for p in reward_net.parameters() if p.requires_grad)
    print("Total number of parameters:", num_total_params)
    print("Number of trainable paramters:", num_trainable_params)

    import torch.optim as optim
    optimizer = optim.Adam(reward_net.parameters(), lr=lr, weight_decay=weight_decay)
    final_weights = learn_reward(device, reward_net, optimizer, train_obs, train_labels, num_epochs, l1_reg,
                                 reward_model_path, val_obs, val_labels, patience, return_weights=return_weights)

    # print out predicted cumulative returns and actual returns
    with torch.no_grad():
        pred_returns = [predict_traj_return(device, reward_net, traj) for traj in sorted_val_demos]
    for i, p in enumerate(pred_returns):
        print(i, p, sorted_val_rewards[i])

    print("train accuracy:", calc_accuracy(device, reward_net, train_obs, train_labels))
    print("validation accuracy:", calc_accuracy(device, reward_net, val_obs, val_labels))
    if test:
        print("test accuracy:", calc_accuracy(device, reward_net, test_obs, test_labels))

    return final_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--feeding', dest='feeding', default=False, action='store_true', help="feeding")
    parser.add_argument('--scratch_itch', dest='scratch_itch', default=False, action='store_true', help="scratch_itch")
    parser.add_argument('--reward_model_path', default='',
                        help="name and location for learned model params, e.g. ./learned_models/breakout.params")
    parser.add_argument('--seed', default=0, type=int, help="random seed for experiments")
    parser.add_argument('--num_comps', default=0, type=int, help="number of pairwise comparisons")
    parser.add_argument('--num_demos', default=120, type=int, help="the number of demos to sample pairwise comps from")
    parser.add_argument('--num_epochs', default=100, type=int, help="number of training epochs")
    parser.add_argument('--hidden_dims', default=0, nargs='+', type=int, help="dimensions of hidden layers")
    parser.add_argument('--lr', default=0.00005, type=float, help="learning rate")
    parser.add_argument('--weight_decay', default=0.0, type=float, help="weight decay")
    parser.add_argument('--l1_reg', default=0.0, type=float, help="l1 regularization")
    parser.add_argument('--patience', default=100, type=int, help="number of iterations we wait before early stopping")
    parser.add_argument('--delta_rank', default=1, type=int, help="min difference between trajectory rankings in our dataset")
    parser.add_argument('--delta_reward', default=0, type=int, help="min difference between trajectory rewards in our dataset")
    parser.add_argument('--all_pairs', dest='all_pairs', default=False, action='store_true', help="whether we generate all pairs from the dataset (num_demos choose 2)")  # NOTE: type=bool doesn't work, value is still true.
    parser.add_argument('--state_action', dest='state_action', default=False, action='store_true', help="whether data consists of state-action pairs rather that just states")  # NOTE: type=bool doesn't work, value is still true.
    parser.add_argument('--augmented', dest='augmented', default=False, action='store_true', help="whether data consists of states + linear features pairs rather that just states")  # NOTE: type=bool doesn't work, value is still true.
    parser.add_argument('--fully_observable', dest='fully_observable', default=False, action='store_true', help="whether data consists of states + (distance, action norm) rather that just states")  # NOTE: type=bool doesn't work, value is still true.
    parser.add_argument('--pure_fully_observable', dest='pure_fully_observable', default=False, action='store_true', help="whether data consists of features that make the preferences fully-observable (without distractor features)")  # NOTE: type=bool doesn't work, value is still true.
    parser.add_argument('--new_pure_fully_observable', dest='new_pure_fully_observable', default=False, action='store_true', help="whether data consists of features that make the preferences fully-observable (without distractor features)")  # NOTE: type=bool doesn't work, value is still true.
    parser.add_argument('--new_fully_observable', dest='new_fully_observable', default=False, action='store_true', help="")  # NOTE: type=bool doesn't work, value is still true.
    parser.add_argument('--num_rawfeatures', default=-1, type=int, help="the number of raw features to keep in the augmented space")
    parser.add_argument('--normalize_features', dest='normalize_features', default=False, action='store_true', help="whether to normalize features")  # NOTE: type=bool doesn't work, value is still true.
    # parser.add_argument('--active_learning', dest='active_learning', default=False, action='store_true', help="whether we use data generated by RL policy's rollouts")  # NOTE: type=bool doesn't work, value is still true.
    parser.add_argument('--teleop', dest='teleop', default=False, action='store_true', help="teleop")
    parser.add_argument('--test', dest='test', default=False, action='store_true', help="testing mode for raw observations")
    args = parser.parse_args()

    seed = args.seed

    ## HYPERPARAMS ##
    feeding = args.feeding
    scratch_itch = args.scratch_itch
    num_comps = args.num_comps
    num_demos = args.num_demos
    hidden_dims = tuple(args.hidden_dims) if args.hidden_dims != 0 else tuple()
    lr = args.lr
    weight_decay = args.weight_decay
    l1_reg = args.l1_reg
    num_epochs = args.num_epochs  # num times through training data
    patience = args.patience
    delta_rank = args.delta_rank
    delta_reward = args.delta_reward
    all_pairs = args.all_pairs
    state_action = args.state_action
    augmented = args.augmented
    fully_observable = args.fully_observable
    pure_fully_observable = args.pure_fully_observable
    new_pure_fully_observable = args.new_pure_fully_observable
    new_fully_observable = args.new_fully_observable
    num_rawfeatures = args.num_rawfeatures
    if num_rawfeatures == -1:
        if feeding:
            num_rawfeatures = 25
        elif scratch_itch:
            num_rawfeatures = 30
        else:
            raise Exception("Need to specify either --feeding or --scratch_itch.")
    normalize_features = args.normalize_features
    # active_learning = args.active_learning
    teleop = args.teleop
    test = args.test
    #################

    run(args.reward_model_path, seed, feeding=feeding, scratch_itch=scratch_itch, num_comps=num_comps, num_demos=num_demos,
        hidden_dims=hidden_dims, lr=lr, weight_decay=weight_decay, l1_reg=l1_reg, num_epochs=num_epochs, patience=patience,
        delta_rank=delta_rank, delta_reward=delta_reward, all_pairs=all_pairs, augmented=augmented, fully_observable=fully_observable,
        pure_fully_observable=pure_fully_observable, new_fully_observable=new_fully_observable, new_pure_fully_observable=new_pure_fully_observable,
        num_rawfeatures=num_rawfeatures, state_action=state_action, normalize_features=normalize_features, teleop=teleop, test=test)

