"""
Script that can be used to play the Unity Reacher environment with a pretrained agent.
"""

import argparse

import numpy as np
import torch
from unityagents import UnityEnvironment

from continous_control.agent import DDPGAgent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-path',
                        dest='env_path',
                        help='Unity environment local path')
    parser.add_argument('--weights-path',
                        dest='weights_path',
                        help='Path to store the agents NN weights',
                        default='./weights')
    args = parser.parse_args()
    return args


def play_with_agent(env: UnityEnvironment, brain_nm: str, agent: DDPGAgent, num_agents: int):
    """
    Uses a pretrained agent to play a Unity environment.

    :param env: Unity environment
    :param brain_nm: brain name
    :param agent: DDPGAgent instance
    :param num_agents: number of Reacher agents
    """
    env_info = env.reset(train_mode=False)[brain_nm]  # reset the environment
    states = env_info.vector_observations  # get the current state for each agent
    score = np.zeros(num_agents)  # initialize the scores for each agent
    while True:
        actions = agent.act(states)  # select action for each agent
        env_info = env.step(actions)[brain_nm]  # send actions to the environment
        next_states, rewards, dones = env_info.vector_observations, env_info.rewards, env_info.local_done
        score += rewards  # update the score
        states = next_states  # roll over the states to next time step
        if any(dones):  # exit loop if episode finished
            break

    print("Score: {}".format(np.mean(score)))


def main():
    args = parse_args()
    # Create env
    unity_env = UnityEnvironment(file_name=args.env_path, seed=10)
    # Get the default brain
    brain_name = unity_env.brain_names[0]
    brain = unity_env.brains[brain_name]
    # Init agent
    environment_info = unity_env.reset(train_mode=False)[brain_name]
    # number of agents
    num_agents = len(environment_info.agents)
    # size of each action
    action_size = brain.vector_action_space_size
    # States
    states = environment_info.vector_observations
    state_size = states.shape[1]
    agent = DDPGAgent(state_size=state_size, action_size=action_size, num_agents=num_agents)
    # Load weights
    agent.actor_local.load_state_dict(torch.load(f'{args.weights_path}/checkpoint_actor.pth'))
    agent.critic_local.load_state_dict(torch.load(f'{args.weights_path}/checkpoint_critic.pth'))
    # Play!
    play_with_agent(unity_env, brain_name, agent, num_agents=num_agents)


if __name__ == '__main__':
    main()
