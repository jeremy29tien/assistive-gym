import assistive_gym
import gym
import pybullet as p
import numpy as np
import csv
import importlib
import argparse

ENV_NAME = "FeedingSawyer-v1"

def make_env(env_name, coop=False, seed=1001):
    if not coop:
        env = gym.make('assistive_gym:'+env_name)
    else:
        module = importlib.import_module('assistive_gym.envs')
        env_class = getattr(module, env_name.split('-')[0] + 'Env')
        env = env_class()
    env.seed(seed)
    return env

def generate_rollout_data(data_dir, seed, num_rollouts, augmented, state_action):
    env = make_env(ENV_NAME, coop=False, seed=seed)
    env.set_seed(seed)  # fixed seed for reproducibility (1000 for training, 1001 for testing)
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

    demos = []
    cum_rewards = []
    for demo in range(num_rollouts):
        traj = []
        cum_reward = 0

        observation = env.reset()
        info = None
        done = False

        start_pos, orient = env.robot.get_pos_orient(env.robot.right_end_effector)
        start_rpy = env.get_euler(orient)
        target_pos_offset = np.zeros(3)
        target_rpy_offset = np.zeros(3)

        # task_success = False
        while not done:
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
            # print("Observation:", observation)
            # print("Action:", action)

            # Handtuned features: spoon-mouth distance, amount of food particles in mouth, amount of food particles on the floor
            distance = np.linalg.norm(observation[7:10])
            if info is None:
                foods_in_mouth = 0
                foods_on_floor = 0
            else:
                foods_in_mouth = info['foods_in_mouth']
                foods_on_floor = info['foods_on_ground']
            handpicked_features = np.array([distance, foods_in_mouth, foods_on_floor])

            if augmented and state_action:
                data = np.concatenate((observation, action, handpicked_features))
            elif augmented:
                data = np.concatenate((observation, handpicked_features))
            elif state_action:
                data = np.concatenate((observation, action))
            else:
                data = observation

            # Step the simulation forward
            observation, reward, done, info = env.step(action)

            traj.append(data)
            cum_reward += reward

        demos.append(traj)
        cum_rewards.append(cum_reward)

    demos = np.asarray(demos)
    cum_rewards = np.asarray(cum_rewards)

    np.save(data_dir+"/demos.npy", demos)
    np.save(data_dir+"/demo_rewards.npy", cum_rewards)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--data_dir', default='', help="location for generated rollouts")
    parser.add_argument('--seed', default=0, type=int, help="random seed for experiments")
    parser.add_argument('--num_rollouts', default=20, type=int, help="number of rollouts")
    parser.add_argument('--state_action', dest='state_action', default=False, action='store_true', help="whether data consists of state-action pairs rather that just states")  # NOTE: type=bool doesn't work, value is still true.
    parser.add_argument('--augmented', dest='augmented', default=False, action='store_true', help="whether data consists of states + linear features pairs rather that just states")  # NOTE: type=bool doesn't work, value is still true.
    args = parser.parse_args()

    data_dir = args.data_dir
    seed = args.seed
    num_rollouts = args.num_rollouts
    state_action = args.state_action
    augmented = args.augmented

    generate_rollout_data(data_dir, seed, num_rollouts, augmented, state_action)
