# Causal Confusion and Reward Misidentification in Preference-Based Reward Learning
_Jeremy Tien, Jerry Zhi-Yang He, Zackory Erickson, Anca D. Dragan, and Daniel S. Brown_

This repository contains the code and data for the **Feeding** and **Itch Scratching** preference learning benchmark environments presented in [**"Causal Confusion and Reward Misidentification in Preference-Based Reward Learning"**](https://openreview.net/pdf?id=R0Xxvr_X3ZA) (**_ICLR 2023_**). 

See the [project website](https://sites.google.com/view/causal-reward-confusion) for supplemental results and videos.
***

## Installing Assistive Gym
We encourage installing in a python virtualenv or conda environment with Python 3.6 or 3.7.
To install, run the following commands in a terminal window: 
```bash
pip3 install --upgrade pip
git clone https://github.com/jeremy29tien/assistive-gym.git
cd assistive-gym
pip3 install -e .
```

You can visualize the various Assistive Gym environments using the environment viewer.  
```bash
python3 -m assistive_gym --env "FeedingSawyer-v1"
```
```bash
python3 -m assistive_gym --env "ScratchItchJaco-v1"
```


## Demonstrations and Pairwise Preference Data
We provide a variety of trajectories and their corresponding rewards for use as demonstrations in preference learning.
Namely, for each environment, we provide:
1. `demos.npy` -- the trajectory data, with shape `(num_trajectories, trajectory_length, observation_dimension)`. (Note: `trajectory_length` is 200 for both Feeding and Itch Scratching.) 
2. `demo_rewards.npy` -- the final cumulative ground truth reward achieved by the corresponding demonstration in `demos.py`. Has shape `(num_trajectories, )`. 
3. `demo_reward_per_timestep.npy` -- the ground truth reward earned by the agent at each timestep in the corresponding demonstration in `demos.npy`. Has shape `(num_trajectories, trajectory_length)`.

The locations of the demonstration data for each environment are:
- Feeding
    - "**Full**" Feature-space (default observation features + add'l. features to make ground-truth reward, TRUE, fully-inferrable): 
        - `assistive-gym/trex/data/feeding/fully_observable/demos.npy`
        - `assistive-gym/trex/data/feeding/fully_observable/demo_rewards.npy`
        - `assistive-gym/trex/data/feeding/fully_observable/demo_reward_per_timestep.npy`
    - "**Pure**" Feature-space ("Full" but with distractor features that are not causal wrt. TRUE removed): 
        - `assistive-gym/trex/data/feeding/pure_fully_observable/demos.npy`
        - `assistive-gym/trex/data/feeding/pure_fully_observable/demos_rewards.npy`
- Itch Scratching
    - "**Full**" Feature-space (default observation features + add'l. features to make ground-truth reward, TRUE, fully-inferrable): 
        - `assistive-gym/trex/data/scratchitch/fully_observable/demos.npy`
        - `assistive-gym/trex/data/scratchitch/fully_observable/demo_rewards.npy`
        - `assistive-gym/trex/data/scratchitch/fully_observable/demo_reward_per_timestep.npy`
    - "**Pure**" Feature-space ("Full" but with distractor features that are not causal wrt. TRUE removed): 
        - `assistive-gym/trex/data/scratchitch/pure_fully_observable/demos.npy`
        - `assistive-gym/trex/data/scratchitch/pure_fully_observable/demo_rewards.npy`
    - "**Full**" Feature-space AND hand-crafted `scratch` feature: 
        - `assistive-gym/trex/data/scratchitch/new_fully_observable/demos.npy`
        - `assistive-gym/trex/data/scratchitch/new_fully_observable/demo_rewards.npy`
        - `assistive-gym/trex/data/scratchitch/new_fully_observable/demo_reward_per_timestep.npy`
    - "**Pure**" Feature-space AND hand-crafted `scratch` feature: 
        - `assistive-gym/trex/data/scratchitch/new_pure_fully_observable/demos.npy`
        - `assistive-gym/trex/data/scratchitch/new_pure_fully_observable/demo_rewards.npy`
        - `assistive-gym/trex/data/scratchitch/new_pure_fully_observable/demo_reward_per_timestep.npy`


To load the data into numpy arrays, one can simply run
```python
demos = np.load("##[DEMOS.NPY PATH]##")
demo_rewards = np.load("##[DEMO_REWARDS.NPY PATH]##")
demo_reward_per_timestep = np.load("##[DEMO_REWARD_PER_TIMESTEP.NPY PATH]##")
```
(where `##[DEMOS.NPY PATH]##` is a path to a `demos.npy` file listed above) within a Python script. 


## Reward Learning from Preferences
We provide `trex/model.py`, a convenient script that loads the trajectory data, creates the pairwise preferences based on the ground truth reward, and performs reward learning on the pairwise preferences. 
To perform reward learning for each of the benchmark environments, run the following in the `assistive-gym/` directory:
- Feeding
    ```bash
    python3 trex/model.py --hidden_dims 128 64 --num_comps 2000 --num_epochs 100 --patience 10 --lr 0.01 --weight_decay 0.01 --seed 0 --reward_model_path ./reward_models/model.params
    ```
- Itch Scratching
    ```bash
    python3 trex/model.py --scratch_itch --hidden_dims 128 64 --num_comps 2000 --num_epochs 100 --patience 10 --lr 0.01 --weight_decay 0.01 --seed $seed --reward_model_path ./reward_models/model.params
    ```
The trained parameters of the reward network will be saved in `assistive-gym/reward_models/model.params`.


## Training the RL Policy
Once the reward network is trained, we can perform reinforcement learning using the preference-learned reward. 
To train, run:
- Feeding
    ```bash
    python3 -m assistive_gym.learn --env "FeedingLearnedRewardSawyer-v0" --algo ppo --seed $seed --train --train-timesteps 1000000 --reward-net-path ./reward_models/model.params --save-dir ./trained_policies/
    ```
- Itch Scratching
    ```bash
    python3 -m assistive_gym.learn --env "ScratchItchLearnedRewardJaco-v0" --algo ppo --seed $seed --train --train-timesteps 1000000 --reward-net-path $reward_model_path --save-dir ./trained_policies/
    ```
 
To evaluate the trained policy on 100 rollouts using the ground truth reward:
- Feeding
    ```bash
      python3 -m assistive_gym.learn --env "FeedingSawyer-v1" --algo ppo --evaluate --eval-episodes 100 --seed 3 --verbose --load-policy-path ./trained_policies/ppo/FeedingLearnedRewardSawyer-v0/checkpoint_000053/checkpoint-53
    ```
- Itch Scratching
    ```bash
      python3 -m assistive_gym.learn --env "ScratchItchJaco-v1" --algo ppo --evaluate --eval-episodes 100 --seed 3 --verbose --load-policy-path ./trained_policies/ppo/ScratchItchLearnedRewardJaco-v0/checkpoint_000053/checkpoint-53
    ```

And to render rollouts of the trained policy:
- Feeding
    ```bash
      python3 -m assistive_gym.learn --env "FeedingSawyer-v1" --algo ppo --render --render-episodes 3 --seed 3 --load-policy-path ./trained_policies/ppo/FeedingLearnedRewardSawyer-v0/checkpoint_000053/checkpoint-53
    ```
- Itch Scratching
    ```bash
      python3 -m assistive_gym.learn --env "ScratchItchJaco-v1" --algo ppo --render --render-episodes 3 --seed 3 --load-policy-path ./trained_policies/ppo/ScratchItchLearnedRewardJaco-v0/checkpoint_000053/checkpoint-53
    ```
