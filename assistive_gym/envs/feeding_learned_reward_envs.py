from .feeding_learned_reward import FeedingLearnedRewardEnv
from .agents import pr2, baxter, sawyer, jaco, stretch, panda, human, human_mesh
from .agents.pr2 import PR2
from .agents.baxter import Baxter
from .agents.sawyer import Sawyer
from .agents.jaco import Jaco
from .agents.stretch import Stretch
from .agents.panda import Panda
from .agents.human import Human
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env

robot_arm = 'right'
human_controllable_joint_indices = human.head_joints
class FeedingLearnedRewardPR2Env(FeedingLearnedRewardEnv):
    def __init__(self):
        super(FeedingLearnedRewardPR2Env, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class FeedingLearnedRewardBaxterEnv(FeedingLearnedRewardEnv):
    def __init__(self):
        super(FeedingLearnedRewardBaxterEnv, self).__init__(robot=Baxter(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class FeedingLearnedRewardSawyerEnv(FeedingLearnedRewardEnv):
    def __init__(self, reward_net_path, indvar=None):
        super(FeedingLearnedRewardSawyerEnv, self).__init__(robot=Sawyer(robot_arm), human=Human(human_controllable_joint_indices, controllable=False), reward_net_path=reward_net_path, indvar=indvar)

class FeedingLearnedRewardJacoEnv(FeedingLearnedRewardEnv):
    def __init__(self):
        super(FeedingLearnedRewardJacoEnv, self).__init__(robot=Jaco(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class FeedingLearnedRewardStretchEnv(FeedingLearnedRewardEnv):
    def __init__(self):
        super(FeedingLearnedRewardStretchEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class FeedingLearnedRewardPandaEnv(FeedingLearnedRewardEnv):
    def __init__(self):
        super(FeedingLearnedRewardPandaEnv, self).__init__(robot=Panda(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class FeedingLearnedRewardPR2HumanEnv(FeedingLearnedRewardEnv, MultiAgentEnv):
    def __init__(self):
        super(FeedingLearnedRewardPR2HumanEnv, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:FeedingLearnedRewardPR2Human-v1', lambda config: FeedingLearnedRewardPR2HumanEnv())

class FeedingLearnedRewardBaxterHumanEnv(FeedingLearnedRewardEnv, MultiAgentEnv):
    def __init__(self):
        super(FeedingLearnedRewardBaxterHumanEnv, self).__init__(robot=Baxter(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:FeedingLearnedRewardBaxterHuman-v1', lambda config: FeedingLearnedRewardBaxterHumanEnv())

class FeedingLearnedRewardSawyerHumanEnv(FeedingLearnedRewardEnv, MultiAgentEnv):
    def __init__(self):
        super(FeedingLearnedRewardSawyerHumanEnv, self).__init__(robot=Sawyer(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:FeedingLearnedRewardSawyerHuman-v1', lambda config: FeedingLearnedRewardSawyerHumanEnv())

class FeedingLearnedRewardJacoHumanEnv(FeedingLearnedRewardEnv, MultiAgentEnv):
    def __init__(self):
        super(FeedingLearnedRewardJacoHumanEnv, self).__init__(robot=Jaco(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:FeedingLearnedRewardJacoHuman-v1', lambda config: FeedingLearnedRewardJacoHumanEnv())

class FeedingLearnedRewardStretchHumanEnv(FeedingLearnedRewardEnv, MultiAgentEnv):
    def __init__(self):
        super(FeedingLearnedRewardStretchHumanEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:FeedingLearnedRewardStretchHuman-v1', lambda config: FeedingLearnedRewardStretchHumanEnv())

class FeedingLearnedRewardPandaHumanEnv(FeedingLearnedRewardEnv, MultiAgentEnv):
    def __init__(self):
        super(FeedingLearnedRewardPandaHumanEnv, self).__init__(robot=Panda(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:FeedingLearnedRewardPandaHuman-v1', lambda config: FeedingLearnedRewardPandaHumanEnv())
