import numpy as np
import random
import copy
from collections import namedtuple, deque

from model_copy import *
from utils import *
from ddpg_agent_copy import *

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
LEARNING_PERIOD = 20    # learning frequency  


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class maddpg_agent():
    """Create a meta-agent that is consisted of 2 DDPG agents which shared a replay buffer """

    def __init__(self, state_size, action_size, num_agents):
        """Initialize meta-agent.
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            num_agents (int): num of agents involved
        """        
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        
        self.agents = [ddpg_agent(state_size, action_size, n+1, random_seed=8) for n in range(num_agents)]
        
        # initiate shared replay memory 
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed=8)
        
    def reset(self):
        for agent in self.agents:
            agent.reset()

    def step(self, states, actions, rewards, next_states, dones, timestep):
        """Save experience in replay buffer, and use random sample from buffer to learn."""
        
        # flattening inputs 
        states = states.reshape(1, -1)                    # 2*24 -- 1*48
        actions = actions.reshape(1, -1)                  # 2*2  -- 1*4
        next_states = next_states.reshape(1, -1)          # 2*24 -- 1*48
        
        # saving experience tuples
        self.memory.add(states, actions, rewards, next_states, dones)
        
        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and timestep % LEARNING_PERIOD == 0:
            for i, agent in enumerate(self.agents):
                experiences = self.memory.sample()
                self.learn(experiences, i)        
    

    def act(self, states, add_noise=False):
        """Picks an action for each agent given."""
        actions = []
        for agent, state in zip(self.agents, states):
            action = agent.act(state, add_noise=True)
            actions.append(action)
        return np.array(actions)    
    
    
    def reset(self):
        """Reset noise for each agent"""
        for agent in self.agents:
            agent.reset()

    def learn(self, experiences, agent_id):
        """
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
        """
        next_actions = []
        actions_pred = []
        states, _, _, next_states, _ = experiences

        # flatten states and next states
        next_states = next_states.reshape(-1, self.num_agents, self.state_size)    # 2*24 -- 1*48
        states = states.reshape(-1, self.num_agents, self.state_size)              # 2*24 -- 1*48
        
        # obtain actions and states for each agent
        for i, agent in enumerate(self.agents):
            agent_id_tensor = self.get_agent_id(i)
            
            state = states.index_select(1, agent_id_tensor).squeeze(1)
            next_state = next_states.index_select(1, agent_id_tensor).squeeze(1)
            
            next_actions.append(agent.actor_target(next_state))
            actions_pred.append(agent.actor_local(state))
            
        next_actions = torch.cat(next_actions, dim=1).to(device)
        actions_pred = torch.cat(actions_pred, dim=1).to(device)
        
        agent = self.agents[agent_id]
        agent.learn(experiences, next_actions, actions_pred)  
        
        
    def get_agent_id(self, i):
        """Convert agent id to a Torch tensor."""
        
        return torch.tensor([i]).to(device)
