import numpy as np
import pybullet as p
import torch

from .env import AssistiveEnv
from .scratch_itch import ScratchItchEnv
from .agents import furniture
from .agents.furniture import Furniture
from trex.model import Net


class ScratchItchLearnedRewardEnv(ScratchItchEnv):
    def __init__(self, robot, human, reward_net_path, indvar):
        super(ScratchItchLearnedRewardEnv, self).__init__(robot=robot, human=human)
        self.augmented = False
        self.state_action = False
        self.num_rawfeatures = 30  # ScratchItchJaco has 30 raw features total
        self.hidden_dims = (128, 64)

        print("reward_net_path:", reward_net_path)
        self.reward_net_path = reward_net_path

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.reward_net = Net("scratch_itch", hidden_dims=self.hidden_dims, with_bias=False, augmented=self.augmented, num_rawfeatures=self.num_rawfeatures, state_action=self.state_action)
        print("device:", self.device)
        print("torch.cuda.is_available():", torch.cuda.is_available())
        self.reward_net.load_state_dict(torch.load(self.reward_net_path, map_location=torch.device('cpu')))
        self.reward_net.to(self.device)

    def step(self, action):
        obs, reward, done, info = super().step(action)

        distance = np.linalg.norm(obs[7:10])
        tool_force_at_target = info['tool_force_at_target']
        handpicked_features = np.array([distance, tool_force_at_target])

        if self.augmented and self.state_action:
            input = np.concatenate((obs, action, handpicked_features))
        elif self.augmented:
            input = np.concatenate((obs[0:self.num_rawfeatures], handpicked_features))
        else:
            input = obs

        # Just modify the reward
        with torch.no_grad():
            reward = self.reward_net.cum_return(torch.from_numpy(np.array([input])).float().to(self.device)).item()

        return obs, reward, done, info
