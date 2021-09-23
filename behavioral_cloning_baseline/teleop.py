import assistive_gym
import gym
import pybullet as p
import numpy as np
import csv
import importlib

FILE_NAME = "data/feedingsawyer_standard.csv"
ENV_NAME = "FeedingSawyer-v1"
ACTIVE_HUMAN = False

def make_env(env_name, coop=False, seed=1001):
    if not coop:
        env = gym.make('assistive_gym:'+env_name)
    else:
        module = importlib.import_module('assistive_gym.envs')
        env_class = getattr(module, env_name.split('-')[0] + 'Env')
        env = env_class()
    env.seed(seed)
    return env


env = make_env(ENV_NAME, coop=ACTIVE_HUMAN, seed=1000)
# env = gym.make('DrinkingSawyer-v1')
env.set_seed(1000)  # fixed seed for reproducibility (1000 for training, 1001 for testing)
env.render()

# Map keys to position and orientation end effector movements
pos_keys_actions = {ord('j'): np.array([-0.01, 0, 0]), ord('l'): np.array([0.01, 0, 0]),
                    ord('u'): np.array([0, -0.01, 0]), ord('o'): np.array([0, 0.01, 0]),
                    ord('k'): np.array([0, 0, -0.01]), ord('i'): np.array([0, 0, 0.01])}
rpy_keys_actions = {ord('k'): np.array([-0.05, 0, 0]), ord('i'): np.array([0.05, 0, 0]),
                    ord('u'): np.array([0, -0.05, 0]), ord('o'): np.array([0, 0.05, 0]),
                    ord('j'): np.array([0, 0, -0.05]), ord('l'): np.array([0, 0, 0.05])}


# Set up file for appending
# file = open(FILE_NAME, 'a+', newline ='')
# write = csv.writer(file)

num_demos = 5

for i in range(num_demos):
    observation = env.reset()

    start_pos, orient = env.robot.get_pos_orient(env.robot.right_end_effector)
    start_rpy = env.get_euler(orient)
    target_pos_offset = np.zeros(3)
    target_rpy_offset = np.zeros(3)

    task_success = False
    while not task_success:
        keys = p.getKeyboardEvents()
        # Process position movement keys ('u', 'i', 'o', 'j', 'k', 'l')
        for key, action in pos_keys_actions.items():
            if p.B3G_SHIFT not in keys and key in keys and keys[key] & p.KEY_IS_DOWN:
                target_pos_offset += action
        # Process rpy movement keys (shift + movement keys)
        for key, action in rpy_keys_actions.items():
            if p.B3G_SHIFT in keys and keys[p.B3G_SHIFT] & p.KEY_IS_DOWN and (key in keys and keys[key] & p.KEY_IS_DOWN):
                target_rpy_offset += action

        # print('Target position offset:', target_pos_offset, 'Target rpy offset:', target_rpy_offset)
        target_pos = start_pos + target_pos_offset
        target_rpy = start_rpy + target_rpy_offset

        # Use inverse kinematics to compute the joint angles for the robot's arm
        # so that its end effector moves to the target position.
        target_joint_angles = env.robot.ik(env.robot.right_end_effector, target_pos, env.get_quaternion(target_rpy), env.robot.right_arm_ik_indices, max_iterations=200, use_current_as_rest=True)
        # Get current joint angles of the robot's arm
        current_joint_angles = env.robot.get_joint_angles(env.robot.right_arm_joint_indices)
        # Compute the action as the difference between target and current joint angles.
        action = (target_joint_angles - current_joint_angles) * 10

        #####################
        ### STANDARD DATA ###
        #####################
        print("Observation:", observation)
        print("Action:", action)
        if ACTIVE_HUMAN:
            data = np.concatenate((observation['robot'], observation['human'], action))
        else:
            data = np.concatenate((observation, action))

        # Write data to file
        print("Data:", data.tolist())
        # write.writerow(data.tolist())

        # Step the simulation forward
        observation, reward, done, info = env.step(action)

        print("Reward:", reward)
        print("Task Success:", info['task_success'])
        print("\n")
        if info['task_success']:
            task_success = True


# file.close()
# print("data appended to", FILE_NAME)
