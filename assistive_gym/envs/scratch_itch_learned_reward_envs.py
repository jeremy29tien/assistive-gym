from .scratch_itch_learned_reward import ScratchItchLearnedRewardEnv
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

robot_arm = 'left'
human_controllable_joint_indices = human.right_arm_joints
class ScratchItchLearnedRewardPR2Env(ScratchItchLearnedRewardEnv):
    def __init__(self):
        super(ScratchItchLearnedRewardPR2Env, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class ScratchItchLearnedRewardBaxterEnv(ScratchItchLearnedRewardEnv):
    def __init__(self):
        super(ScratchItchLearnedRewardBaxterEnv, self).__init__(robot=Baxter(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class ScratchItchLearnedRewardSawyerEnv(ScratchItchLearnedRewardEnv):
    def __init__(self, reward_net_path, indvar=None):
        super(ScratchItchLearnedRewardSawyerEnv, self).__init__(robot=Sawyer(robot_arm), human=Human(human_controllable_joint_indices, controllable=False), reward_net_path=reward_net_path, indvar=indvar)

class ScratchItchLearnedRewardJacoEnv(ScratchItchLearnedRewardEnv):
    def __init__(self, reward_net_path, indvar=None):
        super(ScratchItchLearnedRewardJacoEnv, self).__init__(robot=Jaco(robot_arm), human=Human(human_controllable_joint_indices, controllable=False), reward_net_path=reward_net_path, indvar=indvar)

class ScratchItchLearnedRewardStretchEnv(ScratchItchLearnedRewardEnv):
    def __init__(self):
        super(ScratchItchLearnedRewardStretchEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class ScratchItchLearnedRewardPandaEnv(ScratchItchLearnedRewardEnv):
    def __init__(self):
        super(ScratchItchLearnedRewardPandaEnv, self).__init__(robot=Panda(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class ScratchItchLearnedRewardPR2HumanEnv(ScratchItchLearnedRewardEnv, MultiAgentEnv):
    def __init__(self):
        super(ScratchItchLearnedRewardPR2HumanEnv, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:ScratchItchLearnedRewardPR2Human-v1', lambda config: ScratchItchLearnedRewardPR2HumanEnv())

class ScratchItchLearnedRewardBaxterHumanEnv(ScratchItchLearnedRewardEnv, MultiAgentEnv):
    def __init__(self):
        super(ScratchItchLearnedRewardBaxterHumanEnv, self).__init__(robot=Baxter(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:ScratchItchLearnedRewardBaxterHuman-v1', lambda config: ScratchItchLearnedRewardBaxterHumanEnv())

class ScratchItchLearnedRewardSawyerHumanEnv(ScratchItchLearnedRewardEnv, MultiAgentEnv):
    def __init__(self):
        super(ScratchItchLearnedRewardSawyerHumanEnv, self).__init__(robot=Sawyer(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:ScratchItchLearnedRewardSawyerHuman-v1', lambda config: ScratchItchLearnedRewardSawyerHumanEnv())

class ScratchItchLearnedRewardJacoHumanEnv(ScratchItchLearnedRewardEnv, MultiAgentEnv):
    def __init__(self):
        super(ScratchItchLearnedRewardJacoHumanEnv, self).__init__(robot=Jaco(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:ScratchItchLearnedRewardJacoHuman-v1', lambda config: ScratchItchLearnedRewardJacoHumanEnv())

class ScratchItchLearnedRewardStretchHumanEnv(ScratchItchLearnedRewardEnv, MultiAgentEnv):
    def __init__(self):
        super(ScratchItchLearnedRewardStretchHumanEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:ScratchItchLearnedRewardStretchHuman-v1', lambda config: ScratchItchLearnedRewardStretchHumanEnv())

class ScratchItchLearnedRewardPandaHumanEnv(ScratchItchLearnedRewardEnv, MultiAgentEnv):
    def __init__(self):
        super(ScratchItchLearnedRewardPandaHumanEnv, self).__init__(robot=Panda(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:ScratchItchLearnedRewardPandaHuman-v1', lambda config: ScratchItchLearnedRewardPandaHumanEnv())
