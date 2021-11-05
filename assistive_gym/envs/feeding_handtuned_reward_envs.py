from .feeding_handtuned_reward import FeedingHandtunedRewardEnv
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
class FeedingHandtunedRewardPR2Env(FeedingHandtunedRewardEnv):
    def __init__(self):
        super(FeedingHandtunedRewardPR2Env, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class FeedingHandtunedRewardBaxterEnv(FeedingHandtunedRewardEnv):
    def __init__(self):
        super(FeedingHandtunedRewardBaxterEnv, self).__init__(robot=Baxter(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class FeedingHandtunedRewardSawyerEnv(FeedingHandtunedRewardEnv):
    def __init__(self):
        super(FeedingHandtunedRewardSawyerEnv, self).__init__(robot=Sawyer(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class FeedingHandtunedRewardJacoEnv(FeedingHandtunedRewardEnv):
    def __init__(self):
        super(FeedingHandtunedRewardJacoEnv, self).__init__(robot=Jaco(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class FeedingHandtunedRewardStretchEnv(FeedingHandtunedRewardEnv):
    def __init__(self):
        super(FeedingHandtunedRewardStretchEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class FeedingHandtunedRewardPandaEnv(FeedingHandtunedRewardEnv):
    def __init__(self):
        super(FeedingHandtunedRewardPandaEnv, self).__init__(robot=Panda(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class FeedingHandtunedRewardPR2HumanEnv(FeedingHandtunedRewardEnv, MultiAgentEnv):
    def __init__(self):
        super(FeedingHandtunedRewardPR2HumanEnv, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:FeedingHandtunedRewardPR2Human-v1', lambda config: FeedingHandtunedRewardPR2HumanEnv())

class FeedingHandtunedRewardBaxterHumanEnv(FeedingHandtunedRewardEnv, MultiAgentEnv):
    def __init__(self):
        super(FeedingHandtunedRewardBaxterHumanEnv, self).__init__(robot=Baxter(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:FeedingHandtunedRewardBaxterHuman-v1', lambda config: FeedingHandtunedRewardBaxterHumanEnv())

class FeedingHandtunedRewardSawyerHumanEnv(FeedingHandtunedRewardEnv, MultiAgentEnv):
    def __init__(self):
        super(FeedingHandtunedRewardSawyerHumanEnv, self).__init__(robot=Sawyer(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:FeedingHandtunedRewardSawyerHuman-v1', lambda config: FeedingHandtunedRewardSawyerHumanEnv())

class FeedingHandtunedRewardJacoHumanEnv(FeedingHandtunedRewardEnv, MultiAgentEnv):
    def __init__(self):
        super(FeedingHandtunedRewardJacoHumanEnv, self).__init__(robot=Jaco(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:FeedingHandtunedRewardJacoHuman-v1', lambda config: FeedingHandtunedRewardJacoHumanEnv())

class FeedingHandtunedRewardStretchHumanEnv(FeedingHandtunedRewardEnv, MultiAgentEnv):
    def __init__(self):
        super(FeedingHandtunedRewardStretchHumanEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:FeedingHandtunedRewardStretchHuman-v1', lambda config: FeedingHandtunedRewardStretchHumanEnv())

class FeedingHandtunedRewardPandaHumanEnv(FeedingHandtunedRewardEnv, MultiAgentEnv):
    def __init__(self):
        super(FeedingHandtunedRewardPandaHumanEnv, self).__init__(robot=Panda(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:FeedingHandtunedRewardPandaHuman-v1', lambda config: FeedingHandtunedRewardPandaHumanEnv())
