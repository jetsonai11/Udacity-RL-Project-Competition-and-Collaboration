import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import *
from utils import *

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
EPSILON = 1.0           # epsilon noise parameter
EPSILON_DECAY = 1e-6    # decay parameter of epsilon
LEARNING_PERIOD = 20    # learning frequency  
UPDATE_FACTOR = 10      # how much to learn


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ddpg_agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, agent_id, random_seed=8):
        """Initialize an Agent object.
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.agent_id = agent_id
        self.epsilon = EPSILON

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size).to(device)
        self.actor_target = Actor(state_size, action_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size).to(device)
        self.critic_target = Critic(state_size, action_size).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Make sure target is with the same weight as the source
        self.hard_update(self.actor_target, self.actor_local)
        self.hard_update(self.critic_target, self.critic_local)


    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""

        state = torch.from_numpy(state).float().to(device)

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        if add_noise:
            action += self.epsilon * self.noise.sample()

        return np.clip(action, -1, 1)
  
    
    
    def reset(self):
        """Reset OU Noise"""
        self.noise.reset()

    def learn(self, experiences, next_actions, actions_pred):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(next_state) -> action
            critic_target(next_state, next_action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            next_actions (list): a list of next-state actions for each agent
            actions_pred (list): prediction of actions for each agent at the current state
        """
        states, actions, rewards, next_states, dones = experiences
        agent_id_tensor = torch.tensor([self.agent_id - 1]).to(device)

        # -------------------------------- Update critic -------------------------------- #
        self.critic_optimizer.zero_grad()
        
        # using the target model, compute the next-state Q values
        Q_targets_next = self.critic_target(next_states, next_actions)        
        # compute the current Q target values (y_i) using the Bellman Equation
        Q_targets = rewards.index_select(1, agent_id_tensor) + \
                    (GAMMA * Q_targets_next * (1 - dones.index_select(1, agent_id_tensor)))
        # compute the actual/local Q values using the local model
        Q_expected = self.critic_local(states, actions)
        
        # Minimize critic loss
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        critic_loss.backward()
        self.critic_optimizer.step()

        # -------------------------------- Update actor --------------------------------- #
        self.actor_optimizer.zero_grad()
        
        # Minimize actor loss
        actor_loss = -self.critic_local(states, actions_pred).mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        ### Update target networks with soft updates
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)       

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def hard_update(self, local_model, target_model):
        """Hard update model parameters.
        θ_target = θ_local
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)
            
            
            