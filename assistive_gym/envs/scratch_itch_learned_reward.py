import numpy as np
import pybullet as p
import torch

from .env import AssistiveEnv
from .scratch_itch import ScratchItchEnv
from .agents import furniture
from .agents.furniture import Furniture
from trex.model import Net
from gpu_utils import determine_default_torch_device


class ScratchItchLearnedRewardEnv(ScratchItchEnv):
    def __init__(self, robot, human, reward_net_path, indvar):
        super(ScratchItchLearnedRewardEnv, self).__init__(robot=robot, human=human)

        # Reward Model Specifications
        self.new_pure_fully_observable = False
        self.new_fully_observable = True
        self.pure_fully_observable = False
        self.fully_observable = False
        self.augmented = False
        self.state_action = False
        self.num_rawfeatures = 30  # ScratchItch has 30 raw features total
        self.hidden_dims = (128, 64)
        self.normalize = False

        print("reward_net_path:", reward_net_path)
        self.reward_net_path = reward_net_path

        self.device = torch.device(determine_default_torch_device(not torch.cuda.is_available()))
        self.reward_net = Net("scratch_itch", hidden_dims=self.hidden_dims, augmented=self.augmented, new_pure_fully_observable=self.new_pure_fully_observable, new_fully_observable=self.new_fully_observable, pure_fully_observable=self.pure_fully_observable, fully_observable=self.fully_observable, num_rawfeatures=self.num_rawfeatures, state_action=self.state_action, norm=self.normalize)
        print("device:", self.device)
        print("torch.cuda.is_available():", torch.cuda.is_available())
        self.reward_net.load_state_dict(torch.load(self.reward_net_path, map_location=torch.device('cpu')))
        self.reward_net.to(self.device)

    def step(self, action):
        obs, reward, done, info = super().step(action)

        # Store the ground-truth reward for downstream use (but not training).
        info['gt_reward'] = reward

        distance = np.linalg.norm(obs[7:10])
        tool_force_at_target = info['tool_force_at_target']
        prev_tool_pos_real = info['prev_tool_pos_real']
        robot_force_on_human = info['robot_force_on_human']
        prev_tool_force = info['prev_tool_force']
        scratched = info['scratched']

        handpicked_features = np.array([distance, tool_force_at_target])
        fo_features = np.concatenate((prev_tool_pos_real, [robot_force_on_human, prev_tool_force]))
        new_fo_features = np.concatenate((prev_tool_pos_real, [robot_force_on_human, prev_tool_force, scratched]))

        if self.new_pure_fully_observable:
            input = np.concatenate((obs[0:3], obs[7:10], obs[29:30], action, new_fo_features))
        elif self.new_fully_observable:
            input = np.concatenate((obs, action, new_fo_features))
        elif self.pure_fully_observable:
            input = np.concatenate((obs[0:3], obs[7:10], obs[29:30], action, fo_features))
        elif self.fully_observable:
            input = np.concatenate((obs, action, fo_features))
        elif self.augmented and self.state_action:
            input = np.concatenate((obs, action, handpicked_features))
        elif self.augmented:
            input = np.concatenate((obs[0:self.num_rawfeatures], handpicked_features))
        else:
            input = obs

        # Just modify the reward
        with torch.no_grad():
            reward = self.reward_net.cum_return(torch.from_numpy(np.array([input])).float().to(self.device)).item()

        return obs, reward, done, info
