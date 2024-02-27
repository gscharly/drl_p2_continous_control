# Udacity Deep Reinforcement Learning course - Policy-based methods - P2 Continuous control

This repository contains code that train an agent to solve the environment proposed in the Policy Based Methods section
of the Udacity Deep Reinforcement Learning (DRL) course.

# Environment

![SegmentLocal](./ipynb/agent.gif "segment")

The environment aget is a double-jointed arm that can move to target locations. A reward of +0.1 is provided for each
step that the agent's hand is in the goal location. The goal of the agent is to maintain its position at the target
location for as many time steps as possible. The environment is considered solved when the agent achieves an
average score of 30 or more over 100 consecutive episodes.

Both the action and the state space are continuous. The observation space consists of 33 variables corresponding to
position, rotation, velocity, and angular velocities of the arm. Each action is a vector with 4 numbers,
corresponding to torque applicable to two joints. Every entry in the action vector must be a number between -1 and 1.

There are 2 possible environments. In the first one, there's a single agent, while the second one has 20 agents that
play at the same time.

# Getting started

## Unity environments

Unity doesn't need to be installed since the environment is already available. The environments can be downloaded from
the following links:

Version 1: One agent
- [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
- [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
- [Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
- [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)


Version 2: 20 agents
- [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
- [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- [Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
- [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

## Python dependencies
The project uses Python 3.6 and relies on the [Udacity Value Based Methods repository](https://github.com/udacity/Value-based-methods#dependencies).
This repository should be cloned, and the instructions on the README should be followed to install the necessary
dependencies.

# Instructions
The repository contains 2 scripts under the continuous_control package: train.py and play.py.

## Train
The script train.py can be used to train the agent. The environment has been solved using the Deep Deterministic
Policy Gradient (DDPG) algorithm. More details can be found in ipynb/report.ipynb

The script accepts the following arguments:
- env-path: path pointing to the Unity Reacher environment
- episodes: number of episodes the agent should be trained for
- time-steps-per-episode: timesteps per episode
- weights-path: path where the agent's NN weights will be stored
- learning-rate-actor: Actor learning rate
- learning-rate-critic: Critic learning rate
- weight-decay-actor: Actor NN weight decay rate
- weight-decay-critic: Critic NN weight decay rate
- gamma: discount factor
- batch-size: size of the agent's experience replay buffer
- update-every: update actor & critic after t timesteps
- noise-scalar: scalar that represents the noise to use when altering the actor weights
- noise-scalar-decay: scalar that represents how much should the actor noise increase/decrease in each iteration
- noise-distance: distance between the actor and the noised version of the actor to update the noise scalar

- Example:

```
python train.py --env-path /home/carlos/cursos/udacity_rl_2023/repos/deep-reinforcement-learning/p2_continuous-control/Reacher_Linux_env2/Reacher.x86_64
--weights-path /home/carlos/cursos/udacity_rl_2023/projects/drl_p2_continous_control/weights
-- episodes 300
```

## Play
A trained agent can be used to play! To do so, the play.py script can be used, providing the Unity environment and
the agent's weights paths:

```
python play.py --env-path /home/carlos/cursos/udacity_rl_2023/repos/deep-reinforcement-learning/p2_continuous-control/Reacher_Linux_env2/Reacher.x86_64
--weights-path /home/carlos/cursos/udacity_rl_2023/projects/drl_p2_continous_control/weights
```
