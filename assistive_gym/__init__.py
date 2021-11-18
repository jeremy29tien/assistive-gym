from gym.envs.registration import register

tasks = ['ScratchItch', 'BedBathing', 'Feeding', 'Drinking', 'Dressing', 'ArmManipulation']
robots = ['PR2', 'Jaco', 'Baxter', 'Sawyer', 'Stretch', 'Panda']

for task in tasks:
    for robot in robots:
        register(
            id='%s%s-v1' % (task, robot),
            entry_point='assistive_gym.envs:%s%sEnv' % (task, robot),
            max_episode_steps=200,
        )

for task in ['ScratchItch', 'Feeding']:
    for robot in robots:
        register(
            id='%s%sMesh-v1' % (task, robot),
            entry_point='assistive_gym.envs:%s%sMeshEnv' % (task, robot),
            max_episode_steps=200,
        )

register(
    id='HumanTesting-v1',
    entry_point='assistive_gym.envs:HumanTestingEnv',
    max_episode_steps=200,
)

register(
    id='SMPLXTesting-v1',
    entry_point='assistive_gym.envs:SMPLXTestingEnv',
    max_episode_steps=200,
)

# For our TREX reward learning
register(
    id='FeedingLearnedRewardSawyer-v0',
    entry_point='assistive_gym.envs:FeedingLearnedRewardSawyerEnv',
    max_episode_steps=200,
)

# For testing how well a hand-crafted reward performs (RL-wise)
register(
    id='FeedingHandtunedRewardSawyer-v0',
    entry_point='assistive_gym.envs:FeedingHandtunedRewardSawyerEnv',
    max_episode_steps=200,
)

# For using TREX to learn a linear reward on distance and food info
register(
    id='FeedingLinearRewardSawyer-v0',
    entry_point='assistive_gym.envs:FeedingLinearRewardSawyerEnv',
    max_episode_steps=200,
)
