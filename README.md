# A Study of Causal Confusion in Preference-Based Reward Learning
Jeremy Tien, Jerry Zhi-Yang He, Zackory Erickson, Anca D. Dragan, and Daniel Brown

This repository contains the code and data for the Feeding and Itch Scratching preference learning benchmarks proposed in the paper. 

See the [project website](https://sites.google.com/view/causal-reward-confusion) for supplemental results and videos.
***

## Installing Assistive Gym
We encourage installing in a python virtualenv or conda environment with Python 3.6 or 3.7.
To install, run the following commands in a terminal window: 
```bash
pip3 install --upgrade pip
git clone https://github.com/Healthcare-Robotics/assistive-gym.git
cd assistive-gym
pip3 install -e .
```

## Demonstrations and Pairwise Preference Data
We provide a variety of trajectories and their corresponding rewards for use as demonstrations in preference learning.
Namely, for each environment, we provide:
1. `demos.npy` -- the trajectory data, with shape `(num_trajectories, trajectory_length, observation_dimension)`. (Note: `trajectory_length` is 200 for both Feeding and Itch Scratching.) 
2. `demo_rewards.npy` -- the final cumulative ground truth reward achieved by the corresponding demonstration in `demos.py`. Has shape `(num_trajectories, )`. 
3. `demo_reward_per_timestep.npy` -- the ground truth reward earned by the agent at each timestep in the corresponding demonstration in `demos.npy`. Has shape `(num_trajectories, trajectory_length)`.

The locations of the demonstration data for each environment are:
- Feeding
    - **Raw** Feature-space: 
        - `assistive-gym/trex/data/raw_data/demos_states.npy`
        - `assistive-gym/trex/data/raw_data/demo_rewards.npy`
        - `assistive-gym/trex/data/raw_data/demo_reward_per_timestep.npy`
    - **Augmented** Feature-space: 
        - `assistive-gym/trex/data/augmented_features/demos.npy`
        - `assistive-gym/trex/data/augmented_features/demos_rewards.npy`
        - `assistive-gym/trex/data/augmented_features/demo_reward_per_timestep.npy`
- Itch Scratching
    - **Raw** Feature-space: 
        - `assistive-gym/trex/data/raw_data/demos_states.npy`
        - `assistive-gym/trex/data/raw_data/demo_rewards.npy`
        - `assistive-gym/trex/data/raw_data/demo_reward_per_timestep.npy`
    - **Augmented** Feature-space: 
        - `assistive-gym/trex/data/scratchitch/augmented/demos.npy`
        - `assistive-gym/trex/data/scratchitch/augmented/demos_rewards.npy`
        - `assistive-gym/trex/data/scratchitch/augmented/demo_reward_per_timestep.npy`
        


We provide a [10 Minute Getting Started Guide](https://github.com/Healthcare-Robotics/assistive-gym/wiki/3.-Getting-Started) to help you get familiar with using Assistive Gym for assistive robotics research.

You can visualize the various Assistive Gym environments using the environment viewer.  
A full list of available environment can be found [Here (Environments)](https://github.com/Healthcare-Robotics/assistive-gym/wiki/2.-Environments).
```bash
python3 -m assistive_gym --env "FeedingJaco-v1"
```

We provide pretrained control policies for each robot and assistive task.  
See [Running Pretrained Policies](https://github.com/Healthcare-Robotics/assistive-gym/wiki/4.-Running-Pretrained-Policies) for details on how to run a pretrained policy.

See [Training New Policies](https://github.com/Healthcare-Robotics/assistive-gym/wiki/5.-Training-New-Policies) for documentation on how to train new control policies for Assistive Gym environments.

Finally, [Creating a New Assistive Environment](https://github.com/Healthcare-Robotics/assistive-gym/wiki/6.-Creating-a-New-Assistive-Environment) discusses the process of creating an Assistive Gym environment for your own human-robot interaction tasks.

#### See a list of [common commands available in Assistive Gym ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17Rybu4d2UHIC9D0UA1Au8WSDExX2mMgb?usp=sharing)

## New Features in v1.0
### Clean code syntax
#### v1.0 example (getting robot left end effector velocity)
```python
end_effector_velocity = self.robot.get_velocity(self.robot.left_end_effector)
```
#### Old v0.1 (using default PyBullet syntax)
```python
end_effector_velocity = p.getLinkState(self.robot, 76 if self.robot_type=='pr2' else 19 if self.robot_type=='sawyer' 
                                       else 48 if self.robot_type=='baxter' else 8, computeForwardKinematics=True, 
                                       computeLinkVelocity=True, physicsClientId=self.id)[6]
```

### Google Colab Support
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1PAY5HnLKRB-TBsPaevRr6myMfpVt_yzF?usp=sharing)  
Assistive Gym is now supported in Google Colab! Tons of new examples are now available for developing and learning with Assistive Gym in Google Colab. See the [Wiki-Google Colab](https://github.com/Healthcare-Robotics/assistive-gym/wiki/7.-Google-Colab) for a list of all the available example notebooks.

### Support for mobile bases (mobile manipulation)
For robots with mobile bases, locomotion control is now supported. Ground frictions and slip can be dynamically changed for domain randomization.

Reference this [Google Colab notebook ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1pfYvTcHK1LF8M9p4Gp31S8SziWIiN0Sq?usp=sharing) for an example of mobile base control.  
&nbsp;  
![Mobile bases](images/v1_mobile.gif "Mobile bases")

### Support for the Stretch and PANDA robots
![Stretch](images/v1_stretch.jpg "Stretch")
![PANDA](images/v1_panda.jpg "PANDA")

### Multi-robot control support
Assitive Gym now provides an interface for simulating and controlling multiple robots and people, all through the OpenAI Gym framework. See this example of [multi-robot control ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1NPWZNFpB9NCgTQpbwM78jVHJAC7q_0oR?usp=sharing).  
&nbsp;  
![Multi-robot](images/v1_multi_robot.gif "Multi-robot")

### Integration with iGibson
Assistive Gym can now be used with [iGibson](http://svl.stanford.edu/igibson/) to simulate human-robot interaction in a visually realistic interactive home environment.  
An example of using iGibson with Assistive Gym is available in [this Google Colab notebook ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qFbjuq5lFxPijyw4PFUiZw2sFpXTR7ok?usp=sharing).  
&nbsp;  
![AG iGibson](images/v1_ag_igibson.gif "AG iGibson")

### Static human mesh models (with SMPL-X)
SMPL-X human mesh models are now supported in Assistive Gym. See this [wiki page](https://github.com/Healthcare-Robotics/assistive-gym/wiki/8.-Human-Mesh-Models-with-SMPL-X) for details of how to use these human mesh models.

A Google Colab example of building a simple robot-assisted feeding environment with SMPL-X human meshes is also available: [Assistive Gym with SMPL-X in Colab ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1gz2mQmkTf9g1Jvo6_-WgSQ60cgGHmGOt?usp=sharing)  
&nbsp;  
![SMPL-X human meshes 1](images/v1_smplx_1.jpg "SMPL-X human meshes 1")
![SMPL-X human meshes 2](images/v1_smplx_2.jpg "SMPL-X human meshes 2")

***

## Base Features
### Human and robot models 
Customizable female and male human models (default body sizes and weights matching 50th percentile humans).  
40 actuated human joints (head, torso, arms, waist, and legs)  
&nbsp;  
![Human models](images/human_models.gif "Human models")  
&nbsp;  
Four collaborative robots (PR2, Jaco, Baxter, Sawyer).  
&nbsp;  
![Robot models](images/robot_models.gif "Robot models")
### Realistic human joint limits
Building off of prior research, Assistive Gym provides a model for realistic pose-dependent human joint limits.  
&nbsp;  
![Realistic human joint limits](images/realistic_human_joint_limits.gif "Realistic human joint limits")
### Robot base pose optimization
A robot's base pose can greatly impact the robot’s ability to physically assist people.  
We provide a baseline method using joint-limit-weighted kinematic isotopy (JLWKI) to select good base poses near a person.  
With JLWKI, the robot chooses base poses (position and yaw orientation) with high manipulability near end effector goals.  
&nbsp;  
![Robot base pose optimization](images/robot_base_pose_optimization.gif "Robot base pose optimization")
### Human preferences
During assistance, a person will typically prefer for the robot not to spill water on them, or apply large forces to their body.  
Assistive Gym provides a baseline set of human preferences unified across all tasks, which are incorporated directly into the reward function.
This allows robots to learn to provide assistance that is consist with a person's preferences.  
&nbsp;  
![Human preferences](images/human_preferences.gif "Human preferences")

Refer to [the paper](https://arxiv.org/abs/1910.04700) for details on features in Assistive Gym.
