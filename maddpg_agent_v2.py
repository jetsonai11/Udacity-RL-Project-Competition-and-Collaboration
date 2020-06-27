import copy
import os
import random
from collections import namedtuple, deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import *
from model import *

BUFFER_SIZE = int(1e6)        # replay buffer size
BATCH_SIZE = 256              # minibatch size
GAMMA = 0.99                  # discount factor
TAU = 8e-3                    # soft update parameters
LR_ACTOR = 1e-3               # learning rate of the actor 
LR_CRITIC = 1e-3              # learning rate of the critic
WEIGHT_DECAY = 0              # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    
    """A DDPG agent class that can be initialized based on the number of agents to solve a multi-agent RL environment """

    def __init__(self, state_size, action_size, random_seed):
        """Initialize a maddpg_agent wrapper.
        Params
        ======
            num_agents (int): the number of agents in the environment
            state_size (int): dimension of each state
            action_size (int): dimension of each action
        """
        self.seed = random.seed(random_seed)
        
        self.state_size = state_size
        self.action_size = action_size
                
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), 
                                           lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        
        # OU noise for exploration
        self.noise = OUNoise((1, action_size), random_seed) 
        
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        
        
    def reset(self):
        
        """Resets OU Noise for each agent."""
        
        self.noise.reset()
        
        
    def step(self, state, action, reward, next_state, done, agent_number):
        
        """Save experience in replay memory, and use random sample from buffer to learn."""
        
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        #if len(self.memory) > BATCH_SIZE and timestep % LEARNING_PERIOD == 0:
            #for _ in range(UPDATE_FACTOR):
            
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA, agent_number)        
        
        
    def act(self, states, add_noise=True):
        
        """Returns actions for given state as per current policy."""
        
        states = torch.from_numpy(states).float().to(device)
        actions = np.zeros((1, self.action_size))
        
        self.actor_local.eval()
        
        with torch.no_grad():
            for a_n, state in enumerate(states):
                action = self.actor_local(state).cpu().data.numpy()
                actions[a_n, :] = action
                
        self.actor_local.train()
        
        if add_noise:
            actions += self.noise.sample()
            
        return np.clip(actions, -1, 1)
    
    
    def learn(self, experiences, GAMMA, agent_number):
        
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(next_state) -> action
            critic_target(next_state, next_action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            GAMMA (float): discount factor
            agent_number (int)

        """

        states, actions, rewards, next_states, dones = experiences


        # -------------------------------- Update critic -------------------------------- #
        self.critic_optimizer.zero_grad()
        
        # obtain next-state actions using the target model
        next_actions = self.actor_target(next_states)        
        
        # concat next-state actions for each agent 
        if agent_number == 0:
            next_actions = torch.cat((next_actions, actions[:,2:]), dim=1)
        else:
            next_actions = torch.cat((actions[:,:2], next_actions), dim=1)        
        
        # using the target model, compute the next-state Q values
        Q_targets_next = self.critic_target(next_states, next_actions)        
        # compute the current Q target values (y_i) using the Bellman Equation
        Q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))
        # compute the actual/local Q values using the local model
        Q_expected = self.critic_local(states, actions)
        
        # Minimize critic loss
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # -------------------------------- Update actor --------------------------------- #
        self.actor_optimizer.zero_grad()
        
        # obtain prediction for actions for current states from each agent
        actions_pred = self.actor_local(states)
        
        # concat current-state actions for each agent 
        if agent_number == 0:
            actions_pred = torch.cat((actions_pred, actions[:,2:]), dim=1)
        else:
            actions_pred = torch.cat((actions[:,:2], actions_pred), dim=1) 
            
        # Minimize actor loss
        actor_loss = -self.critic_local(states, actions_pred).mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ------------------- Update target networks with soft updates ------------------ #
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
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

            
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
            
                        
                    