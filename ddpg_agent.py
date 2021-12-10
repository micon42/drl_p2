import numpy as np
import random
from collections import namedtuple, deque
import torch
import torch.nn.functional as F
import torch.optim as optim
from model import ActorNetwork, CriticNetwork
from torchsummary import summary

BATCH_SIZE = 32         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 20       # how often to update the network
NUM_UPDATES = 4         # number of updates
BUFFER_SIZE = 400       # replay buffer size

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


def get_randn(s, min_thresh, max_thresh, m, n):
    while True:
        r = np.reshape(np.random.randn(s), (m, n))
        min_r = np.min(r)
        max_r = np.max(r)
        if min_r > min_thresh and max_r < max_thresh:
            break
    return r


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        np.random.seed(seed)

        # Q-Networks
        self.network_actor_local = ActorNetwork(state_size, action_size, seed).to(device)
        self.network_actor_target = ActorNetwork(state_size, action_size, seed).to(device)
        self.optimizer_actor = optim.Adam(self.network_actor_local.parameters(), lr=LR)

        self.network_critic_local = CriticNetwork(state_size, action_size, seed).to(device)
        self.network_critic_target = CriticNetwork(state_size, action_size, seed).to(device)
        self.optimizer_critic = optim.Adam(self.network_critic_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        # print(self.network_actor_local)
    
    def step(self, states, actions, rewards, next_states, dones):
        # Save experience in replay memory
        for idx, state in enumerate(states):
            self.memory.add(state, actions[idx], rewards[idx], next_states[idx], dones[idx])
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            for upd in range(NUM_UPDATES):
                # If enough samples are available in memory, get random subset and learn
                if len(self.memory) > BATCH_SIZE:
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA)
            # ------------------- update target network ------------------- #
            self.soft_update(self.network_actor_local, self.network_actor_target, TAU)
            self.soft_update(self.network_critic_local, self.network_critic_target, TAU)

    def act(self, states, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        actions = []
        self.network_actor_local.eval()
        for state in states:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action_values = self.network_actor_local(state)

            # Epsilon-greedy action selection
            rnd = np.random.rand()
            if rnd > eps:
                action = action_values.cpu().data.numpy()
            else:
                action = get_randn(self.action_size, -1, 1, 1, 4)
                # if eps > 0.5:
                #     action = get_randn(self.action_size, -1, 1, 1, 4)
                # else:
                #     action = action_values.cpu().data.numpy()
                #     action += get_randn(self.action_size, -0.5, 0.5, 1, 4)
            actions.append(np.clip(action, -1.0, 1.0))
        self.network_actor_local.train()
        return actions

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        action_next = self.network_actor_target(next_states)
        q_targets_next = self.network_critic_target(next_states, action_next)

        # Get expected Q values from local model
        q_expected = self.network_critic_local(states, actions)

        # Compute Q targets for current states
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))

        # Compute loss
        loss_critic = F.mse_loss(q_expected, q_targets).to(device)
        # Minimize the loss
        self.optimizer_critic.zero_grad()
        loss_critic.backward()
        torch.nn.utils.clip_grad_norm(self.network_critic_local.parameters(), 1)
        self.optimizer_critic.step()

        local_actor_prediction = self.network_actor_local(states)
        loss_actor = -self.network_critic_local(states, local_actor_prediction).mean()
        self.optimizer_actor.zero_grad()
        loss_actor.backward()
        self.optimizer_actor.step()

    @staticmethod
    def soft_update(local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)