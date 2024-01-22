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
        self.fc1 = nn.Linear(state_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, action_size)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = self.bn0(state)
        x = F.selu(self.bn1(self.fc1(x)))
        x = F.selu(self.bn2(self.fc2(x)))
        x = F.selu(self.bn3(self.fc3(x)))
        return torch.tanh(self.fc4(x))


class Critic(nn.Module):
    def __init__(self, state_size, action_size, random_seed):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(random_seed)

        self.bn0 = nn.BatchNorm1d(state_size)
        self.fcs1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128 + action_size, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 1)

    def forward(self, state, action):
        state = self.bn0(state)
        x_state = F.selu(self.fcs1(state))
        x = torch.cat((x_state, action), dim=1)
        x = F.selu(self.fc2(x))
        x = F.selu(self.fc3(x))
        x = F.selu(self.fc4(x))
        return F.selu(self.fc5(x))