import torch
import torch.nn as nn
import torch.nn.functional as F


# Adapted from https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-bipedal/model.py

class Actor(nn.Module):

    def __init__(self, state_size: int, action_size: int, random_seed: int):
        """
        Actor (Policy) Model.

        :param state_size: dimension of each state
        :param action_size: dimension of each action
        :param random_seed: random seed
        """

        super(Actor, self).__init__()
        self.seed = torch.manual_seed(random_seed)

        self.bn0 = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Build an actor (policy) network that maps states -> actions.

        :param states: states vector
        :return: actions vector
        """
        x = self.bn0(states)
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        return torch.tanh(self.fc3(x))

    def add_parameter_noise(self, scalar: float = .1):
        """
        Add random noise weighted by a scalar to the layers' weights
        :param scalar: float
        """
        self.fc1.weight.data += torch.randn_like(self.fc1.weight.data) * scalar
        self.fc2.weight.data += torch.randn_like(self.fc2.weight.data) * scalar
        self.fc3.weight.data += torch.randn_like(self.fc3.weight.data) * scalar


class Critic(nn.Module):
    def __init__(self, state_size: int, action_size: int, random_seed: int):
        """
        Critic (Q-value) Model.

        :param state_size: dimension of each state
        :param action_size: dimension of each action
        :param random_seed: random seed
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(random_seed)

        self.bn0 = nn.BatchNorm1d(state_size)
        self.fcs1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256 + action_size, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Build a critic network that maps states -> Q values.

        :param states: states vector
        :param actions: actions vector
        :return: Q value for each pair state-action
        """
        state = self.bn0(states)
        x_state = F.leaky_relu(self.fcs1(state))
        x = torch.cat((x_state, actions), dim=1)
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return F.leaky_relu(self.fc4(x))
