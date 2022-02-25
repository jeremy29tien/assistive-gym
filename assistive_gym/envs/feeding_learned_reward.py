import numpy as np
import pybullet as p
import torch

from .env import AssistiveEnv
from .feeding import FeedingEnv
from .agents import furniture
from .agents.furniture import Furniture
from trex.model import Net


class FeedingLearnedRewardEnv(FeedingEnv):
    # Primus: /home/jtien/assistive-gym/trex/models/5000_trajs_early_stopping.params
    # With weight decay: /home/jtien/assistive-gym/trex/models/5000traj_100epoch_1weightdecay_earlystopping.params
    # With no bias: /home/jtien/assistive-gym/trex/models/5000traj_100epoch_nobias_earlystopping.params
    # Local: /Users/jeremytien/Documents/3rd-Year/Research/Anca Dragan/assistive-gym/trex/models/test1.params
    def __init__(self, robot, human, reward_net_path, indvar):
        super(FeedingLearnedRewardEnv, self).__init__(robot=robot, human=human)
        self.augmented = True
        self.state_action = False
        self.num_rawfeatures = 10
        self.hidden_dims = tuple()

        print("reward_net_path:", reward_net_path)
        self.reward_net_path = reward_net_path

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.reward_net = Net(hidden_dims=self.hidden_dims, with_bias=False, augmented=self.augmented, num_rawfeatures=self.num_rawfeatures, state_action=self.state_action)
        print("device:", self.device)
        print("torch.cuda.is_available():", torch.cuda.is_available())
        self.reward_net.load_state_dict(torch.load(self.reward_net_path, map_location=torch.device('cpu')))
        self.reward_net.to(self.device)

    def step(self, action):
        obs, reward, done, info = super().step(action)

        distance = np.linalg.norm(obs[7:10])  # spoon_pos_real - target_pos_real is at index 7,8,9
        foods_in_mouth = info['foods_in_mouth']
        foods_on_floor = info['foods_on_ground']
        handpicked_features = np.array([distance, foods_in_mouth, foods_on_floor])

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
