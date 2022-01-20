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
    reward_net_path = "/home/jtien/assistive-gym/trex/models/raw_features/60demosallpairs_100epochs_10patience_001lr_01weightdecay_seed2.params"
    reward_net = None
    device = None

    def __init__(self, robot, human):
        super(FeedingLearnedRewardEnv, self).__init__(robot=robot, human=human)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.reward_net = Net(with_bias=False, augmented=True, state_action=False)
        print("device:", self.device)
        print("torch.cuda.is_available():", torch.cuda.is_available())
        self.reward_net.load_state_dict(torch.load(self.reward_net_path, map_location=torch.device('cpu')))
        self.reward_net.to(self.device)

    def step(self, action):
        obs, reward, done, info = super().step(action)

        # If features consist of state-action pairs:
        state = np.concatenate((obs, action))

        # If features consist of just the observation:
        # state = obs

        # Just modify the reward
        with torch.no_grad():
            reward = self.reward_net.cum_return(torch.from_numpy(np.array([state])).float().to(self.device)).item()

        return obs, reward, done, info
