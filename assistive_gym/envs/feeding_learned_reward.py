import numpy as np
import pybullet as p
import torch

from .env import AssistiveEnv
from .feeding import FeedingEnv
from .agents import furniture
from .agents.furniture import Furniture
from trex.model import Net
from gpu_utils import determine_default_torch_device


class FeedingLearnedRewardEnv(FeedingEnv):
    def __init__(self, robot, human, reward_net_path, indvar):
        super(FeedingLearnedRewardEnv, self).__init__(robot=robot, human=human)

        # Reward Model Specifications
        self.pure_fully_observable = False
        self.fully_observable = True
        self.augmented = False
        self.state_action = False
        self.num_rawfeatures = 25  # Feeding has 25 raw features total
        self.hidden_dims = (256, 256, 256)
        self.normalize = False

        print("reward_net_path:", reward_net_path)
        self.reward_net_path = reward_net_path

        self.device = torch.device(determine_default_torch_device(not torch.cuda.is_available()))
        self.reward_net = Net("feeding", hidden_dims=self.hidden_dims, augmented=self.augmented, pure_fully_observable=self.pure_fully_observable, fully_observable=self.fully_observable, num_rawfeatures=self.num_rawfeatures, state_action=self.state_action, norm=self.normalize)
        print("device:", self.device)
        print("torch.cuda.is_available():", torch.cuda.is_available())
        self.reward_net.load_state_dict(torch.load(self.reward_net_path, map_location=torch.device('cpu')))
        self.reward_net.to(self.device)

    def step(self, action):
        obs, reward, done, info = super().step(action)

        # Store the ground-truth reward for downstream use (but not training).
        info['gt_reward'] = reward

        distance = np.linalg.norm(obs[7:10])  # spoon_pos_real - target_pos_real is at index 7,8,9
        foods_in_mouth = info['foods_in_mouth']
        foods_on_floor = info['foods_on_ground']
        foods_hit_human = info['foods_hit_human']
        sum_food_mouth_velocities = info['sum_food_mouth_velocities']
        prev_spoon_pos_real = info['prev_spoon_pos_real']
        robot_force_on_human = info['robot_force_on_human']

        handpicked_features = np.array([distance, foods_in_mouth, foods_on_floor])
        fo_features = np.concatenate(([foods_in_mouth, foods_on_floor, foods_hit_human, sum_food_mouth_velocities],
                                      prev_spoon_pos_real, [robot_force_on_human]))

        if self.pure_fully_observable:
            input = np.concatenate((obs[7:10], obs[24:25], action, fo_features))
        elif self.fully_observable:
            input = np.concatenate((obs, action, fo_features))
        elif self.augmented and self.state_action:
            input = np.concatenate((obs, action, handpicked_features))
        elif self.augmented:
            input = np.concatenate((obs[0:self.num_rawfeatures], handpicked_features))
        elif self.state_action:
            input = np.concatenate((obs, action))
        else:
            input = obs

        # Just modify the reward
        with torch.no_grad():
            reward = self.reward_net.cum_return(torch.from_numpy(np.array([input])).float().to(self.device)).item()

        return obs, reward, done, info
