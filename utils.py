import numpy as np
import gym
import copy
from collections import namedtuple, deque
import random
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class OUNoise:
    """
    Define an Ornstein-Ulhenbeck Process for exploration purpose
    
    """
    def __init__(self, size, scale=1, mu=0, theta=0.15, sigma=0.5):
        """
        Initialize parameters and noise process.
        
        Parameters
        ==========
            mu: long-running mean
            theta: the speed of mean reversion
            sigma: the volatility parameter
            
        """
        self.size = size
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.size) * self.mu
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        
        self.state = np.ones(self.size) * self.mu

    def sample(self):
        """Update internal state and return it as a noise sample."""
        
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        
        return torch.tensor(self.state * self.scale).float()
        
        
class ReplayBuffer:
    """
    Replay buffer for experienced replay

    """

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """
        Initialize a ReplayBuffer object.
        
        Parameters
        ==========
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.experience = namedtuple("Experience", field_names=["state", "action", 
                                                                "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        
        return len(self.memory)
    
    