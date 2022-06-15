import gym
import assistive_gym
import numpy as np
env = gym.make('FeedingSawyer-v1')
env.reset()
for _ in range(5000):
    #env.render()
    observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
    print("obs", observation.shape, observation)
    print("done", done)
    if done:
        break
env.close()
