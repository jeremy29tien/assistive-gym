import numpy as np
import pybullet as p
import torch

from .env import AssistiveEnv
from .feeding import FeedingEnv
from .agents import furniture
from .agents.furniture import Furniture


# This defines a reward function with hand-tuned weights:
# Distance from spoon to mouth: -5
# Food in mouth (in this state): 20
# Food on floor (in this state): -10
class FeedingHandtunedRewardEnv(FeedingEnv):

    def __init__(self, robot, human):
        super(FeedingHandtunedRewardEnv, self).__init__(robot=robot, human=human)
        self.distance_weight = -5
        self.foodmouth_weight = 20
        self.foodfloor_weight = -10

    def step(self, action):
        obs, reward, done, info = super().step(action)
        distance = np.linalg.norm(obs[7:10])  # spoon_pos_real - target_pos_real is at index 7,8,9
        foods_in_mouth = info['foods_in_mouth']
        foods_on_floor = info['foods_on_ground']
        # print("distance", distance)
        # print("foods_in_mouth", foods_in_mouth)
        # print("foods_on_floor", foods_on_floor)

        reward = self.distance_weight*distance + self.foodmouth_weight*foods_in_mouth + self.foodfloor_weight*foods_on_floor
        # print("reward", reward)
        return obs, reward, done, info
