"""
Script that can be used to train a DDPG agent to solve the Reacher simple environment.
"""

import argparse
from collections import deque
import pickle
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from unityagents import UnityEnvironment

from continous_control.agent import DDPGAgent

RANDOM_SEED = 10


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-path',
                        dest='env_path',
                        help='Unity environment local path')
    parser.add_argument('--episodes',
                        dest='episodes',
                        help='Number of episodes to train the agent',
                        default=2000,
                        type=int)
    parser.add_argument('--time-steps-per-episode',
                        dest='max_t',
                        help='Time steps per training episode',
                        default=1000,
                        type=int)
    parser.add_argument('--weights-path',
                        dest='weights_path',
                        help='Path to store the agents NN weights',
                        default='./weights')
    parser.add_argument('--learning-rate-actor',
                        dest='lr_actor',
                        help='Actor NN learning rate',
                        default=1e-4,
                        type=float)
    parser.add_argument('--learning-rate-critic',
                        dest='lr_critic',
                        help='Critic NN learning rate',
                        default=3e-4,
                        type=float)
    parser.add_argument('--weight-decay-actor',
                        dest='weight_decay_actor',
                        help='Actor NN weight decay rate',
                        default=0,
                        type=float)
    parser.add_argument('--weight-decay-critic',
                        dest='weight_decay_critic',
                        help='Critic NN weight decay rate',
                        default=0,
                        type=float)
    parser.add_argument('--gamma',
                        dest='gamma',
                        help='Discount factor',
                        default=0.99,
                        type=float)
    parser.add_argument('--batch-size',
                        dest='batch_size',
                        help='Batch size',
                        default=128,
                        type=int)
    parser.add_argument('--tau',
                        dest='tau',
                        help='Tau',
                        default=5e-3,
                        type=float)
    parser.add_argument('--update-every',
                        dest='update_every',
                        help='Update actor & critic after t timesteps',
                        default=None,
                        type=float)
    parser.add_argument('--noise-scalar',
                        dest='noise_scalar',
                        help='Scalar that represents the noise to use when altering the actor weights',
                        default=None,
                        type=float)
    parser.add_argument('--noise-scalar-decay',
                        dest='noise_scalar_decay',
                        help='Scalar that how much should the actor noise increase/decrease in each iteration',
                        default=.99,
                        type=float)
    parser.add_argument('--noise-distance',
                        dest='noise_distance',
                        help='Distance between the actor and the noised version of the actor to update the noise scalar',
                        default=.1,
                        type=float)
    args = parser.parse_args()
    return args


def plot_scores(scores: List):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


def train_ddpg_complete(env: UnityEnvironment, brain_nm: str, agent: DDPGAgent, weights_path: str,
                        num_agents: int = 20, n_episodes: int = 2000,
                        max_t: int = 1000, n_episodes_score: int = 100):
    scores_deque = deque(maxlen=n_episodes_score)
    scores = []
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_nm]
        states = env_info.vector_observations
        agent.reset()
        score = np.zeros(num_agents)
        for t in range(max_t):
            actions = agent.act(states)
            env_info = env.step(actions)[brain_nm]
            next_states, rewards, dones = env_info.vector_observations, env_info.rewards, env_info.local_done

            agent.step(states, actions, rewards, next_states, dones, timestep=t)
            states = next_states
            score += rewards
            if any(dones):
                break
        scores_deque.append(np.mean(score))
        scores.append(np.mean(score))
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")

        print('\rEpisode {}\t Score: {:.2f}'.format(i_episode, np.mean(score)))
        if i_episode % n_episodes_score == 0:
            torch.save(agent.actor_local.state_dict(), f'{weights_path}/checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), f'{weights_path}/checkpoint_critic.pth')
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

    return scores, agent


def main():
    args = parse_args()
    print(args)
    # Create env
    unity_env = UnityEnvironment(file_name=args.env_path, no_graphics=True)
    # Get the default brain
    brain_name = unity_env.brain_names[0]
    brain = unity_env.brains[brain_name]
    # Init agent
    environment_info = unity_env.reset(train_mode=False)[brain_name]
    # number of agents
    num_agents = len(environment_info.agents)
    # size of each action
    action_size = brain.vector_action_space_size
    # examine the state space
    states = environment_info.vector_observations
    state_size = states.shape[1]
    agent = DDPGAgent(
        state_size=state_size, action_size=action_size,
        random_seed=RANDOM_SEED, lr_actor=args.lr_actor, lr_critic=args.lr_critic,
        weight_decay_actor=args.weight_decay_actor, weight_decay_critic=args.weight_decay_critic,
        gamma=args.gamma, batch_size=args.batch_size, tau=args.tau, update_every=args.update_every,
        num_agents=num_agents, noise_scalar=args.noise_scalar, noise_scalar_decay=args.noise_scalar_decay,
        noise_distance=args.noise_distance
    )
    # Train agent
    result_scores, trained_agent = train_ddpg_complete(
        env=unity_env, brain_nm=brain_name, agent=agent, weights_path=args.weights_path, n_episodes=args.episodes,
        max_t=args.max_t, num_agents=num_agents
    )
    plot_scores(result_scores)

    # Store scores
    with open(f'{args.weights_path}/scores.pkl', 'wb') as f:
        pickle.dump(result_scores, f)


if __name__ == '__main__':
    main()
