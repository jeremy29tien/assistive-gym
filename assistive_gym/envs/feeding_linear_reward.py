import numpy as np
import pybullet as p
import torch

from .env import AssistiveEnv
from .feeding import FeedingEnv
from .agents import furniture
from .agents.furniture import Furniture
from trex.linear_model import Net


class FeedingLinearRewardEnv(FeedingEnv):
    # Primus:
    # With weight decay: /home/jtien/assistive-gym/trex/models/handpicked/5000traj_1epoch_1weightdecay_earlystopping.params
    # Without weight decay: /home/jtien/assistive-gym/trex/models/handpicked/5000traj_1epoch_noweightdecay_earlystopping.params
    # Local: /Users/jeremytien/Documents/3rd-Year/Research/Anca Dragan/assistive-gym/trex/models/handpicked/5000traj_1epoch_1weightdecay_earlystopping.params

    # Stress test:
    # /home/jtien/assistive-gym/trex/models/linear/6demosallpairs_100epochs_1weightdecay.params
    # /home/jtien/assistive-gym/trex/models/linear/5demosallpairs_10epochs_001lr_01weightdecay_seedX.params
    def __init__(self, robot, human):
        super(FeedingLinearRewardEnv, self).__init__(robot=robot, human=human)
        self.augmented = True
        self.state_action = False
        self.num_rawfeatures = 1

        self.reward_net_path = "/home/jtien/assistive-gym/trex/models/linear/augmented_1770comps_60pairdelta_100epochs_10patience_001lr_01weightdecay_seed0.params"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.reward_net = Net(augmented=self.augmented, num_rawfeatures=self.num_rawfeatures, state_action=self.state_action)
        print("device:", self.device)
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
            input = handpicked_features

        # Just modify the reward
        with torch.no_grad():
            reward = self.reward_net.cum_return(torch.from_numpy(np.array([input])).float().to(self.device)).item()

        return obs, reward, done, info
