import trex.model
import assistive_gym.learn
import argparse
import numpy as np
import multiprocessing, ray
import re, string
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F



# Function similar to active_learn's get_rollouts to get rollouts from trained policy
# env_name: ScratchItchJaco-v1 or FeedingSawyer-v1
def get_rollouts(env_name, num_rollouts, policy_path, seed, pure_fully_observable=False, fully_observable=False):
    ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False)
    # Set up the environment
    env = assistive_gym.learn.make_env(env_name, seed=seed)
    # Load pretrained policy from file
    test_agent, _ = assistive_gym.learn.load_policy(env, 'ppo', env_name, policy_path, seed=seed)

    new_rollouts = []
    new_rollout_rewards = []
    for r in range(num_rollouts):
        traj = []
        reward_total = 0.0
        obs = env.reset()
        info = None
        done = False
        while not done:
            action = test_agent.compute_action(obs)

            # FeedingSawyer
            # augmented (privileged) features: spoon-mouth distance, amount of food particles in mouth, amount of food particles on the floor
            # fully-observable: add previous end effector position, robot force on human, food information
            if env_name == "FeedingSawyer-v1":
                distance = np.linalg.norm(obs[7:10])
                if info is None:
                    foods_in_mouth = 0
                    foods_on_floor = 0
                    foods_hit_human = 0
                    sum_food_mouth_velocities = 0
                    prev_spoon_pos_real = np.zeros(3)
                    robot_force_on_human = 0
                else:
                    foods_in_mouth = info['foods_in_mouth']
                    foods_on_floor = info['foods_on_ground']
                    foods_hit_human = info['foods_hit_human']
                    sum_food_mouth_velocities = info['sum_food_mouth_velocities']
                    prev_spoon_pos_real = info['prev_spoon_pos_real']
                    robot_force_on_human = info['robot_force_on_human']
                privileged_features = np.array([distance, foods_in_mouth, foods_on_floor])
                fo_features = np.concatenate(([foods_in_mouth, foods_on_floor, foods_hit_human,
                                               sum_food_mouth_velocities], prev_spoon_pos_real, [robot_force_on_human]))
                # Features from the raw observation that are causal:
                # spoon_pos_real - target_pos_real and self.spoon_force_on_human, respectively
                pure_obs = np.concatenate((obs[7:10], obs[24:25]))

            # ScratchItchJaco privileged features: end effector - target distance, total force at target
            if env_name == "ScratchItchJaco-v1":
                distance = np.linalg.norm(obs[7:10])
                if info is None:
                    tool_force_at_target = 0.0
                    prev_tool_pos_real = np.zeros(3)
                    robot_force_on_human = 0
                    prev_tool_force = 0
                else:
                    tool_force_at_target = info['tool_force_at_target']
                    prev_tool_pos_real = info['prev_tool_pos_real']
                    robot_force_on_human = info['robot_force_on_human']
                    prev_tool_force = info['prev_tool_force']
                privileged_features = np.array([distance, tool_force_at_target])
                fo_features = np.concatenate((prev_tool_pos_real, [robot_force_on_human, prev_tool_force]))
                # Features from the raw observation that are causal:
                # tool_pos_real, tool_pos_real - target_pos_real, and self.tool_force, respectively
                pure_obs = np.concatenate((obs[0:3], obs[7:10], obs[29:30]))

            if pure_fully_observable:
                data = np.concatenate((pure_obs, action, fo_features))
            elif fully_observable:
                data = np.concatenate((obs, action, fo_features))
            else:
                data = obs

            obs, reward, done, info = env.step(action)

            traj.append(data)
            reward_total += reward

        new_rollouts.append(traj)
        new_rollout_rewards.append(reward_total)

    new_rollouts = np.asarray(new_rollouts)
    new_rollout_rewards = np.asarray(new_rollout_rewards)
    return new_rollouts, new_rollout_rewards


# Function that prepares the training data:
# - Squishes the rollouts / trajectories into just a bunch of states
# - Adds labels of 0 for reward-learning and 1 for policy training.
# Dimensions of reward_learning_trajs and policy_trajs: (num_trajs, traj_length, feature_dim)
def prepare_data(reward_learning_trajs, policy_trajs):
    reward_learning_obs = np.reshape(reward_learning_trajs, (reward_learning_trajs.shape[0] * reward_learning_trajs.shape[1], reward_learning_trajs.shape[2]))
    policy_obs = np.reshape(policy_trajs, (policy_trajs.shape[0] * policy_trajs.shape[1], policy_trajs.shape[2]))
    reward_learning_labels = np.zeros((reward_learning_obs.shape[0], 1))
    policy_labels = np.ones((policy_obs.shape[0], 1))
    reward_learning_data = np.concatenate((reward_learning_obs, reward_learning_labels), axis=-1)
    policy_data = np.concatenate((policy_obs, policy_labels), axis=-1)

    data = np.concatenate((reward_learning_data, policy_data), axis=0)
    np.random.shuffle(data)
    return data[:, :-1], data[:, -1]


# Define discriminator model that classifies between states from reward-learning training data (0)
# and states from the trained policy (1) by minimizing cross-entropy loss.
class Discriminator(nn.Module):
    def __init__(self, env_name, hidden_dims=(128, 64), augmented=False, fully_observable=False, pure_fully_observable=False, new_fully_observable=False, new_pure_fully_observable=False, num_rawfeatures=25, state_action=False, norm=False):
        super().__init__()

        if new_pure_fully_observable:
            if env_name == "FeedingSawyer-v1":
                raise Exception("NOT IMPLEMENTED.")
            elif env_name == "ScratchItchJaco-v1":
                input_dim = 20
        if new_fully_observable:
            if env_name == "FeedingSawyer-v1":
                raise Exception("NOT IMPLEMENTED.")
            elif env_name == "ScratchItchJaco-v1":
                input_dim = 43
        elif pure_fully_observable:
            if env_name == "FeedingSawyer-v1":
                input_dim = 19
            elif env_name == "ScratchItchJaco-v1":
                input_dim = 19
        elif fully_observable:
            if env_name == "FeedingSawyer-v1":
                input_dim = 40
            elif env_name == "ScratchItchJaco-v1":
                input_dim = 42
        elif augmented and state_action:
            # Feeding only
            input_dim = 35
        elif augmented:
            if env_name == "FeedingSawyer-v1":
                input_dim = num_rawfeatures + 3
            elif env_name == "ScratchItchJaco-v1":
                input_dim = num_rawfeatures + 2
        elif state_action:
            # Feeding only
            input_dim = 32
        else:
            if env_name == "FeedingSawyer-v1":
                input_dim = 25
            elif env_name == "ScratchItchJaco-v1":
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

    def forward(self, x):
        # Normalize features
        if self.normalize:
            x = self.layer_norm(x)

        for l in range(self.num_layers - 1):
            x = F.leaky_relu(self.fcs[l](x))
        logit = self.fcs[-1](x)
        return logit


def train(device, discriminator_network, optimizer, training_inputs, training_outputs, num_epochs, l1_reg, checkpoint_dir, val_obs, val_labels, patience, return_weights=False):
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
            obs = np.array(training_obs[i])
            label = np.array([training_labels[i]])
            obs = torch.from_numpy(obs).float().to(device)
            label = torch.from_numpy(label).to(device)

            # zero out gradient
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = discriminator_network.forward(obs)
            # outputs = outputs.unsqueeze(0)
            # print("train outputs", outputs.shape)
            # print("train label", label.shape)

            # Calculate loss
            cross_entropy_loss = loss_criterion(outputs, label)
            l1_loss = l1_reg * torch.linalg.vector_norm(torch.cat([param.view(-1) for param in discriminator_network.parameters()]), 1)
            loss = cross_entropy_loss + l1_loss

            # Backpropagate
            loss.backward()

            # Take one optimizer step
            optimizer.step()

        val_loss = calc_val_loss(device, discriminator_network, val_obs, val_labels)
        val_acc = calc_accuracy(device, discriminator_network, val_obs, val_labels)
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
            torch.save(discriminator_network.state_dict(), checkpoint_dir)
            if return_weights:
                print("Weights:", discriminator_network.state_dict())
                final_weights = discriminator_network.state_dict()

        prev_min_val_loss = min(prev_min_val_loss, val_loss)
    print("Finished training.")
    if return_weights:
        return final_weights
    return None


# Calculates the cross-entropy losses over the entire validation set and returns the MEAN.
def calc_val_loss(device, discriminator_network, training_inputs, training_outputs):
    loss_criterion = nn.CrossEntropyLoss()
    losses = []
    with torch.no_grad():
        for i in range(len(training_inputs)):
            obs = np.array(training_inputs[i])
            label = np.array([training_outputs[i]])
            obs = torch.from_numpy(obs).float().to(device)
            label = torch.from_numpy(label).to(device)

            # Forward to get logits
            outputs = discriminator_network.forward(obs)
            # outputs = outputs.unsqueeze(0)
            # print("val outputs", outputs.shape)
            # print("val label", label.shape)

            loss = loss_criterion(outputs, label)
            losses.append(loss.item())

    return np.mean(losses)


def calc_accuracy(device, discriminator_network, training_inputs, training_outputs):
    num_correct = 0.
    with torch.no_grad():
        for i in range(len(training_inputs)):
            obs = np.array(training_inputs[i])
            label = training_outputs[i]
            obs = torch.from_numpy(obs).float().to(device)

            # Forward to get logits
            outputs = discriminator_network.forward(obs)
            _, pred_label = torch.max(outputs, 0)
            if pred_label.item() == label:
                num_correct += 1.
    return num_correct / len(training_inputs)


def get_logit(device, net, x):
    rewards_from_obs = []
    with torch.no_grad():
        logit = net.forward(torch.from_numpy(x).float().to(device)).item()
    return logit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--seed', default=0, type=int, help="random seed for experiments")
    # Have an argument for the reward-learning training data file
    # Have an argument for the trained policy to pull rollouts from
    # Have an argument for how many trajectories / rollouts to pull from both data pools

    args = parser.parse_args()


    seed = args.seed
    np.random.seed(seed)

    # Load data used in reward-learning

    # Get rollouts from trained policy

    # Prepare discriminator training data

    # Train discriminator

    # Report training and validation accuracy

    # Save model. In the future, load model if exists.

    # Inference:
    # For the states visited during reward-learning, calculate the average return/logit. --> $D_{KL}(p(x) || q(x))$

    # For the states visited during the policy rollout, calculate the average return/logit and NEGATE. --> $D_{KL}(q(x) || p(x))$
    pass
