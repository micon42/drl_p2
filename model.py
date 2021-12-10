import torch
import torch.nn as nn
import torch.nn.functional as F 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BaseNetwork(nn.Module):

    def __init__(self, state_size, seed, fc1_units):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(BaseNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc_phi1 = nn.Linear(state_size, fc1_units)
        self.bn_phi = nn.BatchNorm1d(fc1_units)

        self.to(device)

    def forward(self, state, action=None):
        p = F.relu(self.fc_phi1(state))
        p = self.bn_phi(p)
        return p


class ActorNetwork(BaseNetwork):
    """Actor Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        base_units = 512
        super(ActorNetwork, self).__init__(state_size, seed, fc1_units=base_units)
        self.seed = torch.manual_seed(seed)
        self.fc1_a = nn.Linear(base_units, fc1_units)
        self.fc_actor = nn.Linear(fc1_units, action_size)
        self.to(device)

    def forward(self, state):
        base = super().forward(state)
        a = F.relu(self.fc1_a(base))
        a = torch.tanh(self.fc_actor(a))
        return a


class CriticNetwork(BaseNetwork):
    """Critic Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        base_units = 512
        super(CriticNetwork, self).__init__(state_size, seed, fc1_units=base_units)
        self.seed = torch.manual_seed(seed)
        self.fc1_c = nn.Linear(base_units+action_size, fc1_units)
        self.fc_critic = nn.Linear(fc1_units, action_size)
        self.to(device)

    def forward(self, state, action):
        base = super().forward(state, action)
        c = F.relu(self.fc1_c(torch.cat([base, action], dim=1)))
        c = self.fc_critic(c)
        return c
