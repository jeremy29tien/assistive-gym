from .feeding_linear_reward import FeedingLinearRewardEnv
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
class FeedingLinearRewardPR2Env(FeedingLinearRewardEnv):
    def __init__(self):
        super(FeedingLinearRewardPR2Env, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class FeedingLinearRewardBaxterEnv(FeedingLinearRewardEnv):
    def __init__(self):
        super(FeedingLinearRewardBaxterEnv, self).__init__(robot=Baxter(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class FeedingLinearRewardSawyerEnv(FeedingLinearRewardEnv):
    def __init__(self):
        super(FeedingLinearRewardSawyerEnv, self).__init__(robot=Sawyer(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class FeedingLinearRewardJacoEnv(FeedingLinearRewardEnv):
    def __init__(self):
        super(FeedingLinearRewardJacoEnv, self).__init__(robot=Jaco(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class FeedingLinearRewardStretchEnv(FeedingLinearRewardEnv):
    def __init__(self):
        super(FeedingLinearRewardStretchEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class FeedingLinearRewardPandaEnv(FeedingLinearRewardEnv):
    def __init__(self):
        super(FeedingLinearRewardPandaEnv, self).__init__(robot=Panda(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class FeedingLinearRewardPR2HumanEnv(FeedingLinearRewardEnv, MultiAgentEnv):
    def __init__(self):
        super(FeedingLinearRewardPR2HumanEnv, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:FeedingLinearRewardPR2Human-v1', lambda config: FeedingLinearRewardPR2HumanEnv())

class FeedingLinearRewardBaxterHumanEnv(FeedingLinearRewardEnv, MultiAgentEnv):
    def __init__(self):
        super(FeedingLinearRewardBaxterHumanEnv, self).__init__(robot=Baxter(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:FeedingLinearRewardBaxterHuman-v1', lambda config: FeedingLinearRewardBaxterHumanEnv())

class FeedingLinearRewardSawyerHumanEnv(FeedingLinearRewardEnv, MultiAgentEnv):
    def __init__(self):
        super(FeedingLinearRewardSawyerHumanEnv, self).__init__(robot=Sawyer(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:FeedingLinearRewardSawyerHuman-v1', lambda config: FeedingLinearRewardSawyerHumanEnv())

class FeedingLinearRewardJacoHumanEnv(FeedingLinearRewardEnv, MultiAgentEnv):
    def __init__(self):
        super(FeedingLinearRewardJacoHumanEnv, self).__init__(robot=Jaco(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:FeedingLinearRewardJacoHuman-v1', lambda config: FeedingLinearRewardJacoHumanEnv())

class FeedingLinearRewardStretchHumanEnv(FeedingLinearRewardEnv, MultiAgentEnv):
    def __init__(self):
        super(FeedingLinearRewardStretchHumanEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:FeedingLinearRewardStretchHuman-v1', lambda config: FeedingLinearRewardStretchHumanEnv())

class FeedingLinearRewardPandaHumanEnv(FeedingLinearRewardEnv, MultiAgentEnv):
    def __init__(self):
        super(FeedingLinearRewardPandaHumanEnv, self).__init__(robot=Panda(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:FeedingLinearRewardPandaHuman-v1', lambda config: FeedingLinearRewardPandaHumanEnv())
