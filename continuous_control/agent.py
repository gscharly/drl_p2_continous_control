from collections import namedtuple, deque
import copy
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from continuous_control.model import Actor, Critic

BUFFER_SIZE = int(1e5)  # replay buffer size
SIGMA_DECAY = 0.95  # Noise decay
SIGMA_MIN = 0.005

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Adapted from https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-bipedal/ddpg_agent.py
class DDPGAgent:
    def __init__(self, state_size: int, action_size: int, num_agents: int = 20, random_seed: int = 10,
                 lr_actor: float = 1e-4,
                 lr_critic: float = 3e-4, weight_decay_actor: float = .0, weight_decay_critic: float = .0,
                 gamma: float = 0.99, batch_size: int = 128, tau: float = 1e-3, update_every: int = None,
                 noise_scalar: float = None, noise_scalar_decay: float = .99, noise_distance: float = None
                 ):
        """
        Agent that implements the DDPG algorithm

        :param state_size: dimension of each state
        :param action_size: dimension of each action
        :param num_agents: number of agents in the environment. Defaults to 20
        :param random_seed: random seed
        :param lr_actor: actor NN learning rate
        :param lr_critic: critic NN learning rate
        :param weight_decay_actor: actor NN weight decay rate
        :param weight_decay_critic: critic NN weight decay rate
        :param gamma: discount rate
        :param batch_size: number of samples to train the NNs with
        :param tau: target NNs soft update param
        :param update_every: how many iterations to train the NNs between. Defaults to None, meaning that the NNs will
        be updated at every t
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.update_every = update_every
        self.num_agents = num_agents
        self.noise_scalar = noise_scalar
        self.noise_scalar_decay = noise_scalar_decay
        self.noise_distance = noise_distance
        random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(), lr=lr_actor,
                                                weight_decay=weight_decay_actor)
        if self.noise_scalar is not None:
            self.actor_noised = Actor(state_size, action_size, random_seed).to(device)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(), lr=lr_critic,
                                                 weight_decay=weight_decay_critic)

        # Noise process used to encourage exploration
        self.noise = OUNoise(action_size, num_agents=num_agents, seed=random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, self.batch_size, random_seed)

    def step(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_states: np.ndarray,
             dones: np.ndarray,
             timestep: int = 0):
        """
        Save experience in replay memory, and use random sample from buffer to learn.

        :param states: state vector
        :param actions: action vector
        :param rewards: reward
        :param next_states: next state vector
        :param dones: integer indicating whether the task has finished
        :param timestep: timestep iteration in an episode
        """
        # Save experience / reward
        for i in range(self.num_agents):
            self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])

        # Learn, if enough samples are available in memory
        learn_cond = len(self.memory) > self.batch_size
        if self.update_every is not None:
            learn_cond = learn_cond and timestep % self.update_every == 0

        if learn_cond:
            experiences = self.memory.sample()
            self.learn(experiences, self.gamma)

    def act(self, states: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Returns actions for given state as per current policy.

        :param states: states vector
        :param add_noise: whether to add noise to encourage exploration
        :return: actions vector
        """
        states = torch.from_numpy(states).float().to(device)
        self.actor_local.eval()
        # Adaptive noise scaling from https://soeren-kirchner.medium.com/deep-deterministic-policy-gradient-ddpg-with-and-without-ornstein-uhlenbeck-process-e6d272adfc3
        with torch.no_grad():
            # get the action values from the noised actor for comparison
            actions = self.actor_local(states).cpu().data.numpy()
            if add_noise and self.noise_scalar is not None:
                # hard copy the actor_regular to actor_noised
                self.actor_noised.load_state_dict(self.actor_local.state_dict().copy())
                # add noise to the copy
                self.actor_noised.add_parameter_noise(self.noise_scalar)
                # get the next action values from the noised actor
                actions_noised = self.actor_noised(states).cpu().data.numpy()
                # measure the distance between the action values from the regular and
                # the noised actor to adjust the amount of noise that will be added next round
                distance = np.sqrt(np.mean(np.square(actions - actions_noised)))

                # adjust the amount of noise given to the actor_noised
                if distance > self.noise_distance:
                    self.noise_scalar *= self.noise_scalar_decay
                if distance <= self.noise_distance:
                    self.noise_scalar /= self.noise_scalar_decay

        self.actor_local.train()
        if add_noise and self.noise_scalar is None:
            actions += self.noise.sample()
        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences: Tuple, gamma: float):
        """
        Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        :param experiences: tuple of (s, a, r, s', done) tuples
        :param gamma: discount factor
        """

        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Gradient clipping
        # torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

    @staticmethod
    def soft_update(local_model: torch.nn.Module, target_model: torch.nn.Module, tau: float):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        :param local_model: PyTorch model (weights will be copied from)
        :param target_model: PyTorch model (weights will be copied to)
        :param tau: interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class OUNoise:

    def __init__(self, size: int, num_agents: int, seed: int = 10, mu: float = 0., theta: float = 0.1, sigma: float = 0.1,
                 sigma_decay: float = None):
        """
        Ornstein-Uhlenbeck process.

        :param size: size of the resulting vector
        :param seed: random seed
        :param mu: noise mean
        :param theta: theta param
        :param sigma: noise variance
        """
        self.mu = mu * np.ones((num_agents, size))
        self.theta = theta
        self.sigma = sigma
        self.sigma_decay = sigma_decay
        random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
        if self.sigma_decay is not None and self.sigma > SIGMA_MIN:
            self.sigma *= self.sigma_decay

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * (np.random.rand(*x.shape)-0.5)
        self.state = x + dx
        return self.state


class ReplayBuffer:

    def __init__(self, action_size: int, buffer_size: int, batch_size: int, seed: int = 10):
        """
        Fixed-size buffer to store experience tuples.

        :param action_size: dimension of each action
        :param buffer_size: maximum size of buffer
        :param batch_size:  size of each training batch
        :param seed: random seed
        """

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        random.seed(seed)

    def add(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: int):
        """
        Add a new experience to memory.

        :param state: state vector
        :param action: action vector
        :param reward: reward
        :param next_state: next state vector
        :param done: integer indicating whether the task has finished
        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self) -> Tuple:
        """
        Randomly sample a batch of experiences from memory.
        :return: experience tuple
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
