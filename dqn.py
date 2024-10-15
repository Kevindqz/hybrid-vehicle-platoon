import time
from typing import Any
from dmpcpwa.mpc.mpc_mld import MpcMld
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dmpcpwa.agents.mld_agent import MldAgent
from mpcs.rl_mpc import RlMpcAgent, RlMpcCent
from gymnasium import Env
from env import PlatoonEnv
import numpy as np
from misc.common_controller_params import Params, Sim
from gymnasium.wrappers import TimeLimit
from mpcrl.wrappers.envs import MonitorEpisodes
from models import Vehicle, Platoon
import pickle
from fleet_cent_mld import MpcGearCent
from plot_fleet import plot_fleet
from fleet_cent_mld import TrackingCentralizedAgent
import wandb
#set seed
# np.random.seed(19)
# random.seed(1)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    """A cyclic buffer of bounded size that holds the transitions observed recently."""

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
 
class DRQN(nn.Module):
    """A class that defines a Deep Recurrent Q-Network (DRQN) model."""

    def __init__(self, input_size, hidden_size, num_actions = 3, num_layers=1): # three actions : downshift, no shift, upshift
        super(DRQN, self).__init__()

        # self.fc1 = nn.Linear(input_size, hidden_size*4)
        # self.rnn = nn.GRU(hidden_size*4, hidden_size, num_layers, batch_first=True, bidirectional=True)
        # self.fc2 = nn.Linear(hidden_size*2, num_actions)

        self.fc = nn.Linear(hidden_size*2, num_actions)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
    def forward(self, x):
        # fc_out = self.fc1(x)      
        # drqn_out, _ = self.rnn(fc_out)
        # q_values = self.fc2(drqn_out)

        drqn_out, _ = self.rnn(x)
        q_values = self.fc(drqn_out)
        return q_values

class DqnAgent():
    """A class that set up RL training scheme with DQN Algorithm and interfaces with Mpc."""
    def __init__(self, batch_size, gamma, eps_start, eps_end, eps_decay, target_update, learning_rate, device)-> None:
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.EPS_START = eps_start
        self.EPS_END = eps_end
        self.EPS_DECAY = eps_decay
        self.tau = target_update
        self.LEARNING_RATE = learning_rate
        self.device = device

        self.n_layers = 2
        self.n_states = 4 # 3 states: position, velocity, gear, 1 action
        # self.n_states = 3 # 3 states: velocity, gear, 1 action
        self.n_hidden = 64
        self.n_actions = 3
        self.n_gears = 6
        self.pred_horizon = 5
        self.ts = Params.ts
        self.policy_net = DRQN(self.n_states, self.n_hidden, self.n_actions, self.n_layers).to(self.device)
        self.target_net = DRQN(self.n_states, self.n_hidden, self.n_actions, self.n_layers).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.LEARNING_RATE, amsgrad=True)
        self.memory_size = 100000
        self.memory = ReplayMemory(self.memory_size)
        self.steps_done = 0
        self.episode_durations = []
        self.episode_rewards = []
        self.tracking_costs = []
        self.fuel_costs = []
        self.penalties = []
        self.gear_violations = []
        self.infeasibilities = []
        self.performance = []
        self.runtimes = []

        self.fuel_param = 1
        self.tracking_param = 0.001

    def select_action(self, state, mode):
        if mode == 'train':
            eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                math.exp(-1. * self.steps_done / self.EPS_DECAY)
        elif mode == 'evaluate':
            eps_threshold = 0
        else:
            raise ValueError("Mode should be either 'train' or 'evaluate'")
        # eps_threshold = 0 
        self.steps_done += 1
        sample = random.random()
        if sample > eps_threshold: 
            with torch.no_grad():
                q_values = self.policy_net(state)
                actions = q_values.argmax(2)
        else:  #this random action does not make sense
            actions = torch.tensor([random.randrange(0,self.n_actions) for _ in range(state.size(1))], device=self.device, dtype=torch.long)
            actions = actions.unsqueeze(0)
        
        # calculate the gear choice by time step and avoid gear violation      
        # actions = torch.zeros((1, self.pred_horizon), device=self.device, dtype=torch.int64)
        # current_gear = state[0, 0, 3]
        # hidden = None 
        # if sample > eps_threshold:
        #     with torch.no_grad():
        #         for i in range(self.pred_horizon):
        #             # feed the state to network one by one
        #             q_values, hidden = self.policy_net(state[:, i, :].unsqueeze(1), hidden)
        #             if current_gear == 6: # if current gear is 6, upshift is not allowed
        #                 q_values[0, 0, 2] = -float('inf')
        #             elif current_gear == 1: # if current gear is 1, downshift is not allowed
        #                 q_values[0, 0, 0] = -float('inf')
        #             actions[0, i] = q_values.argmax(2)
        #             current_gear = current_gear + actions[0, i].item() - 1
        # else:
        #     for i in range(self.pred_horizon):
        #         if current_gear == 6:
        #             actions[0, i] = random.choice([0, 1])
        #         elif current_gear == 1:
        #             actions[0, i] = random.choice([1, 2])
        #         else:
        #             actions[0, i] = random.randrange(0, self.n_actions)
        #         current_gear = current_gear + actions[0, i].item() - 1
        return actions

    def plot_rewards(self, show_result=False):        
        plt.figure(2)
        rewards_t = torch.tensor(self.episode_rewards, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Total Cost')
        plt.plot(rewards_t.numpy())
        if len(rewards_t) >= 100:
            means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((rewards_t.cumsum(0)[:99] / torch.arange(1, 100), means))
        else:
            means =  rewards_t.cumsum(0) / torch.arange(1, len(rewards_t) + 1)

        plt.plot(means.numpy())
        plt.savefig(f'cost_fuel_{self.fuel_param}_tracking_{self.tracking_param}.png')
        plt.pause(0.001)    
    
    def plot_gear_violation(self, show_result=False):
        plt.figure(3)
        penalties_t = torch.tensor(self.gear_violations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Gear Violations')
        plt.plot(penalties_t.numpy())
        # if len(penalties_t) >= 100:
        #     means = penalties_t.unfold(0, 100, 1).mean(1).view(-1)
        #     means = torch.cat((penalties_t.cumsum(0)[:99] / torch.arange(1, 100), means))
        # else:
        #     means = penalties_t.cumsum(0) / torch.arange(1, len(penalties_t) + 1)
        # plt.plot(means.numpy())
        plt.savefig(f'gear_violation_fuel_{self.fuel_param}_tracking_{self.tracking_param}.png')
        plt.pause(0.001)

    def plot_infeasibility(self, show_result=False):
        plt.figure(4)
        penalties_t = torch.tensor(self.infeasibilities, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Infeasibilities')
        plt.plot(penalties_t.numpy())
        # if len(penalties_t) >= 100:
        #     means = penalties_t.unfold(0, 100, 1).mean(1).view(-1)
        #     means = torch.cat((penalties_t.cumsum(0)[:99] / torch.arange(1, 100), means))
        # else:
        #     means = penalties_t.cumsum(0) / torch.arange(1, len(penalties_t) + 1)
        # plt.plot(means.numpy())
        plt.savefig(f'infeasibility_fuel_{self.fuel_param}_tracking_{self.tracking_param}.png')
        plt.pause(0.001)

    def plot_tracking_cost(self, show_result=False):
        fig = plt.figure(5)
        penalties_t = torch.tensor(self.tracking_costs, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Tracking Cost')
        plt.plot(penalties_t.numpy())
        if len(penalties_t) >= 100:
            means = penalties_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((penalties_t.cumsum(0)[:99] / torch.arange(1, 100), means))
        else:
            means = penalties_t.cumsum(0) / torch.arange(1, len(penalties_t) + 1)

        plt.plot(means.numpy())
        plt.savefig(f'tracking_cost_fuel_{self.fuel_param}_tracking_{self.tracking_param}.png')
        plt.pause(0.001)
        return fig
    
    def plot_fuel_cost(self, show_result=False):   
        plt.figure(6)
        penalties_t = torch.tensor(self.fuel_costs, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Fuel Cost')
        plt.plot(penalties_t.numpy())
        if len(penalties_t) >= 100:
            means = penalties_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((penalties_t.cumsum(0)[:99] / torch.arange(1, 100), means))
        else:
            means = penalties_t.cumsum(0) / torch.arange(1, len(penalties_t) + 1)

        plt.plot(means.numpy())
        plt.savefig(f'fuel_cost_fuel_{self.fuel_param}_tracking_{self.tracking_param}.png')
        plt.pause(0.001)
    
    def plot_penalties(self, show_result=False):
        plt.figure(7)
        penalties_t = torch.tensor(self.penalties, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Penalties')
        plt.plot(penalties_t.numpy())
        if len(penalties_t) >= 100:
            means = penalties_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((penalties_t.cumsum(0)[:99] / torch.arange(1, 100), means))
        else:
            means = penalties_t.cumsum(0) / torch.arange(1, len(penalties_t) + 1)

        plt.plot(means.numpy())
        plt.savefig(f'penalties_fuel_{self.fuel_param}_tracking_{self.tracking_param}.png')
        plt.pause(0.001)

    def optimize_model(self):
        """Perform one step of the optimization (on the policy network)"""

        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch. This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state) 
        action_batch = torch.cat(batch.action)
        reward_batch = - torch.cat(batch.reward) # original is cost

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values  = self.policy_net(state_batch).gather(2, action_batch.unsqueeze(2))

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, self.pred_horizon, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(2).values

        # Compute the expected Q values (Only first step of each sequence is used)
        # expected_state_action_values = (next_state_values[:,0] * self.GAMMA) + reward_batch
        
        # Compute the expected Q values, extend the reward to the length of the sequence
        reward_batch = reward_batch.unsqueeze(-1)
        reward_batch = reward_batch.unsqueeze(-1)
        reward_batch_extended = reward_batch.expand(-1, self.pred_horizon, -1)

        # Compute the expected Q values (All steps of each sequence are used)
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch_extended.squeeze(-1)
        # Compute Huber loss``
        criterion = nn.SmoothL1Loss()
        # loss = criterion(state_action_values[:,0,:], expected_state_action_values.unsqueeze(-1))
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(-1))
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def train(self, env: PlatoonEnv, mpc_agent: RlMpcAgent, num_episodes: int, seed: int, save: bool = True):
        """
        This method trains the DQN agent with the given environment and Mpc agent.
        """

        wandb.init(project="DRQN-MPC",
                config={"architecture": "rnn",
                        "num_rnn_layers": self.n_layers,
                        "hidden_size": self.n_hidden,
                        "episodes": num_episodes,
                        "learning_rate": self.LEARNING_RATE, 
                        "gamma": self.GAMMA, 
                        "batch_size": self.BATCH_SIZE, 
                        "target_update": self.tau,
                        "tracking_param": self.tracking_param,
                        "fuel_param": self.fuel_param,
                        "eps_start": self.EPS_START,
                        "eps_end": self.EPS_END,
                        "memory_size": self.memory_size,
                        "seed": seed
                        })
                   
        seeds = map(int, np.random.SeedSequence(seed).generate_state(num_episodes))
        for episode, current_seed in zip(range(num_episodes), seeds):
            # Initialize the environment and get its state
            state, info = env.reset(seed = current_seed)
            # state, _ = env.reset(seed = 23)
            # get new reference trajectory
            mpc_agent.leader_x = info['leader_trajectory']

            mpc_agent.on_episode_start(env, episode, state) 

            # Initialize the Mpc agent and get its state，solved by mppc
            vehicle = env.platoon.get_vehicles()
            j = vehicle[0].get_gear_from_velocity(state[1,0].item())

            #assume all time step have the same gear choice
            gear_choice_init_explicit = np.ones((self.pred_horizon, 1)) * j
            gear_choice_init_binary = np.zeros((self.n_gears, self.pred_horizon))

            #set value for gear choice binary
            for i in range(self.pred_horizon):
                gear_choice_init_binary[int(gear_choice_init_explicit[i])-1, i] = 1

            # solve mpc with initial gear choice
            for i in range(len(Vehicle.b)):
                for k in range(mpc_agent.mpc.N):
                    mpc_agent.mpc.sigma[i, 0, k].ub = gear_choice_init_binary[i, k]
                    mpc_agent.mpc.sigma[i, 0, k].lb = gear_choice_init_binary[i, k]

            u, info = mpc_agent.get_control(state) 
            pred_u = torch.tensor(info['u'].T, dtype=torch.float32, device=self.device).unsqueeze(0)
            pred_x = torch.tensor(info['x'][:,:-1].T, dtype=torch.float32, device=self.device).unsqueeze(0)
            pred_vel = pred_x[[0], :, [1]]
            pred_pos = pred_x[[0], :, [0]]

            # Get the reference trajectory
            ref_pos = torch.tensor(mpc_agent.leader_x[0, 0 : mpc_agent.N], dtype=torch.float32, device=self.device).unsqueeze(0)
            
            # Calculate the relative position
            rel_pos = pred_pos - ref_pos

            # Normalize the velocity prediction
            pred_vel_norm = (pred_vel - env.platoon.vehicles[0].vl[0]) / (env.platoon.vehicles[0].vh[-1] - env.platoon.vehicles[0].vl[0])
            
            pred_states = torch.cat((rel_pos.unsqueeze(-1), pred_vel_norm.unsqueeze(-1), pred_u), 2)
            # pred_states = torch.cat((pred_vel_norm.unsqueeze(-1), pred_u), 2) # only use velocity and input as state
            last_gear_choice_explicit = gear_choice_init_explicit

            total_reward = 0
            total_tracking_cost = 0
            total_fuel_cost = 0
            total_penalty = 0
            gear_violation_count = 0
            infeasibility_count = 0
            timestep = 0

            # Step the environment
            observation, _ , terminated, truncated, rewards = env.step(u)
            mpc_agent.on_env_step(env, episode, timestep)
            tracking_cost = self.tracking_param * float(rewards['cost_tracking'])
            fuel_cost = self.fuel_param * float(rewards['cost_fuel'])
            total_reward += tracking_cost + fuel_cost
            total_tracking_cost += self.tracking_param * float(rewards['cost_tracking'])
            total_fuel_cost += self.fuel_param * float(rewards['cost_fuel'])

            timestep += 1

            # Set new reference trajectory
            mpc_agent.on_timestep_end(env, episode, timestep)

            while not (terminated or truncated):
                
                penalty = 0

                action = self.select_action(pred_states, "train") # state_seq: (batch_size, seq_length, input_size)

                gear_shift = action - 1
                gear_shift = gear_shift.cpu().numpy()
                gear_shift = gear_shift.T
                
                # Transform action to gear choice 
                # doesnt make sense, should shift based on previous time step
                # gear_choice_explicit = last_gear_choice_explicit + gear_shift.T 
                gear_choice_explicit = np.ones((self.pred_horizon, 1))
                gear_choice_explicit[0, 0] = last_gear_choice_explicit[0, 0] + gear_shift[0, 0]
                for i in range(1, self.pred_horizon):
                   gear_choice_explicit[i, 0] = gear_choice_explicit[i-1, 0] + gear_shift[i, 0]


                # Check if gear choice is out of range，if so, use gear choice of last time step, give large penalty
                for i in range(self.pred_horizon):
                    if gear_choice_explicit[i] < 1:
                        gear_choice_explicit[i] = 1
                        penalty += 100
                        gear_violation_count += 1
                    elif gear_choice_explicit[i] > 6:
                        gear_choice_explicit[i] = 6
                        penalty += 100
                        gear_violation_count += 1

                gear_choice_binary = np.zeros((self.n_gears, self.pred_horizon))
                for i in range(self.pred_horizon):
                    gear_choice_binary[int(gear_choice_explicit[i])-1, i] = 1

                #Interface with Mpc agent
                for i in range(len(Vehicle.b)):
                    for k in range(mpc_agent.mpc.N):
                        mpc_agent.mpc.sigma[i, 0, k].ub = gear_choice_binary[i, k]
                        mpc_agent.mpc.sigma[i, 0, k].lb = gear_choice_binary[i, k]     

                u, info = mpc_agent.get_control(observation, try_again_if_infeasible=False)

                #check if mpc is infeasible, if so, use shifted prediction and action from last time step 
                if info["cost"] == float('inf'):
                    infeasibility_count += 1
                    penalty += 500
                    gear_choice_explicit = np.concatenate((last_gear_choice_explicit[1:], last_gear_choice_explicit[[-1]]),0)
                    gear_choice_binary = np.zeros((self.n_gears, self.pred_horizon))
                    for i in range(self.pred_horizon):
                        gear_choice_binary[int(gear_choice_explicit[i])-1, i] = 1

                    if not np.all(np.sum(gear_choice_binary, axis=0) == 1):
                        raise ValueError("gear binaries not correctly set!") 
                    o = 5
                    for i in range(len(Vehicle.b)):
                        for k in range(mpc_agent.mpc.N):
                            mpc_agent.mpc.sigma[i, 0, k].ub = gear_choice_binary[i, k] 
                            mpc_agent.mpc.sigma[i, 0, k].lb = gear_choice_binary[i, k]
                        # for k in range(o, mpc_agent.mpc.N):
                        #     mpc_agent.mpc.sigma[i, 0, k].ub = 1
                        #     mpc_agent.mpc.sigma[i, 0, k].lb = 0
                        
                    u, info = mpc_agent.get_control(observation, try_again_if_infeasible=False)     

                    if info["cost"] == float('inf'):
                        j = vehicle[0].get_gear_from_velocity(observation[1].item())
            
                        #assume all time step have the same gear choice
                        gear_choice_init_explicit = np.ones((self.pred_horizon, 1)) * j
                        gear_choice_init_binary = np.zeros((self.n_gears, self.pred_horizon))
                        #set value for gear choice binary
                        for i in range(self.pred_horizon):
                            gear_choice_init_binary[int(gear_choice_init_explicit[i])-1, i] = 1

                        for i in range(len(Vehicle.b)):
                            for k in range(mpc_agent.mpc.N):
                                mpc_agent.mpc.sigma[i, 0, k].ub = gear_choice_init_binary[i, k]
                                mpc_agent.mpc.sigma[i, 0, k].lb = gear_choice_init_binary[i, k]

                        gear_choice_explicit = gear_choice_init_explicit

                        u, info = mpc_agent.get_control(observation, try_again_if_infeasible=False) 

                        if info['cost'] == float('inf'):
                            raise RuntimeError("Backup gear solution was still infeasible, reconsider theory. Oh no.")            


                observation, _ , terminated, truncated, rewards = env.step(u)
                mpc_agent.on_env_step(env, episode, timestep)


                fuel_cost = self.fuel_param * torch.tensor([rewards['cost_fuel']], device=self.device) #only care about fuel
                tracking_cost =self.tracking_param * torch.tensor([rewards['cost_tracking']], device=self.device)
                penalty = torch.tensor([penalty], device=self.device)
                total_reward += float(fuel_cost) + float(tracking_cost) + float(penalty)
                # total_reward += float(tracking_cost) + float(penalty)
                current_reward = penalty + tracking_cost + fuel_cost
                # current_reward = penalty + 0.001 * tracking_cost
                total_tracking_cost += float(tracking_cost)
                total_fuel_cost += float(fuel_cost)
                total_penalty += float(penalty)

                done = terminated or truncated
                if terminated:
                    next_pred_states = None
                else:
                    pred_u_next = info['u']
                    pred_x_next = info['x']
                    # transform the predicition to tensor
                    pred_u_next = torch.tensor(info['u'].T, dtype=torch.float32, device=self.device).unsqueeze(0)
                    pred_x_next = torch.tensor(info['x'][:,:-1].T, dtype=torch.float32, device=self.device).unsqueeze(0)
                    pred_vel_next = pred_x_next[[0], :, 1]
                    pred_pos_next = pred_x_next[[0], :, 0]

                    # Get the reference trajectory
                    ref_pos_next = torch.tensor(mpc_agent.leader_x[0, timestep : timestep + mpc_agent.N], dtype=torch.float32, device=self.device).unsqueeze(0)

                    # Calculate the relative position
                    rel_pos_next = pred_pos_next - ref_pos_next

                    # Normalize the velocity prediction
                    pred_vel_next_norm = (pred_vel_next - env.platoon.vehicles[0].vl[0]) / (env.platoon.vehicles[0].vh[-1] - env.platoon.vehicles[0].vl[0])

                    next_pred_states = torch.cat((rel_pos_next.unsqueeze(-1), pred_vel_next_norm.unsqueeze(-1), pred_u_next), 2)
                    # next_pred_states = torch.cat((pred_vel_next_norm.unsqueeze(-1), pred_u_next), 2)
                    

                # Store the transition in memory
                self.memory.push(pred_states, action, next_pred_states, current_reward) 

                # Move to the next state
                pred_states = next_pred_states
                last_gear_choice_explicit = gear_choice_explicit

                timestep += 1
                mpc_agent.on_timestep_end(env, episode, timestep)
                
                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    self.episode_rewards.append(total_reward)
                    self.gear_violations.append(gear_violation_count)
                    self.infeasibilities.append(infeasibility_count)
                    self.tracking_costs.append(total_tracking_cost)
                    self.fuel_costs.append(total_fuel_cost)
                    self.penalties.append(total_penalty)
                    # self.plot_rewards()
                    # self.plot_gear_violation()
                    # self.plot_infeasibility()
                    print(f"Episode {episode}, tracking cost: {total_tracking_cost}, fuel cost: {total_fuel_cost}, penalty: {total_penalty}, gear violation: {gear_violation_count}, infeasibility: {infeasibility_count}, total cost: {total_reward}")

                    wandb.log({"tracking_cost": total_tracking_cost, 
                               "fuel_cost": total_fuel_cost, 
                               "penalty": total_penalty, 
                               "total_cost": total_reward, 
                               "gear_violations": gear_violation_count, 
                               "infeasibilities": infeasibility_count})
                    
                    mpc_agent.on_episode_end(env, episode, total_reward)
                    break

                # add checkpoints for every 10000 episodes
                # if i_episode % 10000 == 0:
                #     self.save_model(f'checkpoint_{i_episode}.pt')
                   
        print('Complete')
        wandb.finish()
        # Use a dictionary to store all the data
        train_data = {}

        # X = list(env.observations)
        # U = list(env.actions)
        # R = list(env.rewards)

        train_data['states'] = list(env.observations)
        train_data['actions'] = list(env.actions)
        train_data['rewards'] = list(env.rewards)
        train_data['episode_rewards'] = self.episode_rewards
        train_data['gear_violations'] = self.gear_violations
        train_data['infeasibilities'] = self.infeasibilities
        train_data['tracking_costs'] = self.tracking_costs
        train_data['fuel_costs'] = self.fuel_costs
        train_data['penalties'] = self.penalties

        # Save the data
        with open('train_data.pkl', 'wb') as f:
            pickle.dump(train_data, f)

        env.close()

        if save:
            self.save_model('trained_dqn_agent_full_complexity_best_setup.pth')

        # Plot the results and save the figures
        self.plot_rewards()
        self.plot_gear_violation()
        self.plot_infeasibility()
        self.plot_tracking_cost()
        self.plot_fuel_cost()
        self.plot_penalties()     
        plt.ioff()
        plt.show()

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path))

    def evaluate(self, env: PlatoonEnv, 
                 mpc_agent: TrackingCentralizedAgent, 
                 num_episodes: int, 
                 seed: int = None,
                 open_loop: bool = False,
                 save: bool = False,
                 plot: bool = False,
        ):
        self.policy_net.eval()
        returns = np.zeros(num_episodes)
        mpc_agent.on_validation_start(env)
        seeds = map(int, np.random.SeedSequence(seed).generate_state(num_episodes))

        if num_episodes > 1:
            wandb.init(project="DRQN-MPC-EVAL",
                config={"architecture": "rnn",
                        "episodes": num_episodes,
                        "num_rnn_layers": self.n_layers,
                        "hidden_size": self.n_hidden,
                        "learning_rate": self.LEARNING_RATE, 
                        "gamma": self.GAMMA, 
                        "batch_size": self.BATCH_SIZE, 
                        "target_update": self.tau,
                        "tracking_param": self.tracking_param,
                        "fuel_param": self.fuel_param,
                        })
        
        for episode, current_seed in zip(range(num_episodes), seeds):
            start_time = time.time()

            total_runtime = 0
            total_tracking_cost = 0
            total_fuel_cost = 0
            mpc_agent.reset(current_seed)
            # state, _ = env.reset(seed=current_seed, options=env_reset_options)
            state, info = env.reset(seed = current_seed)
            truncated, terminated, timestep = False, False, 0
            
            mpc_agent.leader_x = info['leader_trajectory']

            mpc_agent.on_episode_start(env, episode, state)

            # Initialize the Mpc agent and get its state，solved by mppc
            vehicle = env.platoon.get_vehicles()
            j = vehicle[0].get_gear_from_velocity(state[1,0].item())

            #assume all time step have the same gear choice
            gear_choice_init_explicit = np.ones((self.pred_horizon, 1)) * j
            gear_choice_init_binary = np.zeros((self.n_gears, self.pred_horizon))

            #set value for gear choice binary
            for i in range(self.pred_horizon):
                gear_choice_init_binary[int(gear_choice_init_explicit[i])-1, i] = 1

            # solve mpc with initial gear choice
            for i in range(len(Vehicle.b)):
                for k in range(mpc_agent.mpc.N):
                    mpc_agent.mpc.sigma[i, 0, k].ub = gear_choice_init_binary[i, k]
                    mpc_agent.mpc.sigma[i, 0, k].lb = gear_choice_init_binary[i, k]

            u, info = mpc_agent.get_control(state) 
            pred_u = torch.tensor(info['u'].T, dtype=torch.float32, device=self.device).unsqueeze(0)
            pred_x = torch.tensor(info['x'][:,:-1].T, dtype=torch.float32, device=self.device).unsqueeze(0)
            pred_vel = pred_x[[0], :, [1]]
            pred_pos = pred_x[[0], :, [0]]

            # Get the reference trajectory
            ref_pos = torch.tensor(mpc_agent.leader_x[0, 0 : mpc_agent.N], dtype=torch.float32, device=self.device).unsqueeze(0)
            
            # Calculate the relative position
            rel_pos = pred_pos - ref_pos

            # Normalize the velocity prediction
            pred_vel_norm = (pred_vel - env.platoon.vehicles[0].vl[0]) / (env.platoon.vehicles[0].vh[-1] - env.platoon.vehicles[0].vl[0])
            
            pred_states = torch.cat((rel_pos.unsqueeze(-1), pred_vel_norm.unsqueeze(-1), pred_u), 2)

            last_gear_choice_explicit = gear_choice_init_explicit

            total_reward = 0
            gear_violation_count = 0
            infeasibility_count = 0
            performance = 0
            gear_violation = [0]
            infeasibility = [0]

            timestep = 0

            # Step the environment
            state, r , terminated, truncated, rewards = env.step(u)
            mpc_agent.on_env_step(env, episode, timestep)
            returns[episode] += r
            total_tracking_cost += rewards['cost_tracking']
            total_fuel_cost += rewards['cost_fuel']
            total_reward += rewards['cost_tracking'] + rewards['cost_fuel']
            timestep += 1
            mpc_agent.on_timestep_end(env, episode, timestep)

            if open_loop:
                _, info = mpc_agent.get_control(state)
                actions = info["u"]
                counter = 0

            while not (truncated or terminated):
                # changed origonal agents evaluate here to use the mld mpc
                if not open_loop:
                    penalty = 0

                    action = self.select_action(pred_states, "evaluate") # state_seq: (batch_size, seq_length, input_size)

                    gear_shift = action - 1
                    gear_shift = gear_shift.cpu().numpy()
                    gear_shift = gear_shift.T
                    
                    # Transform action to gear choice 
                    # doesnt make sense, should shift based on previous time step
                    # gear_choice_explicit = last_gear_choice_explicit + gear_shift.T 
                    gear_choice_explicit = np.ones((self.pred_horizon, 1))
                    gear_choice_explicit[0, 0] = last_gear_choice_explicit[0, 0] + gear_shift[0, 0]
                    for i in range(1, self.pred_horizon):
                        gear_choice_explicit[i, 0] = gear_choice_explicit[i-1, 0] + gear_shift[i, 0]


                    # Check if gear choice is out of range，if so, use gear choice of last time step, give large penalty
                    count = 0
                    for i in range(self.pred_horizon):
                        if gear_choice_explicit[i] < 1:
                            gear_choice_explicit[i] = 1
                            penalty += 100
                            gear_violation_count += 1
                            count += 1
                        elif gear_choice_explicit[i] > 6:
                            gear_choice_explicit[i] = 6
                            penalty += 100
                            gear_violation_count += 1
                            count += 1
                    gear_violation.append(count)

                    gear_choice_binary = np.zeros((self.n_gears, self.pred_horizon))
                    for i in range(self.pred_horizon):
                        gear_choice_binary[int(gear_choice_explicit[i])-1, i] = 1

                    #Interface with Mpc agent
                    for i in range(len(Vehicle.b)):
                        for k in range(mpc_agent.mpc.N):
                            mpc_agent.mpc.sigma[i, 0, k].ub = gear_choice_binary[i, k]
                            mpc_agent.mpc.sigma[i, 0, k].lb = gear_choice_binary[i, k]     

                    u, info = mpc_agent.get_control(state, try_again_if_infeasible=False)

                    #check if mpc is infeasible, if so, use shifted prediction and action from last time step 
                    count = 0
                    if info["cost"] == float('inf'):
                        infeasibility_count += 1
                        count = 1
                        penalty += 500
                        gear_choice_explicit = np.concatenate((last_gear_choice_explicit[1:], last_gear_choice_explicit[[-1]]),0)
                        gear_choice_binary = np.zeros((self.n_gears, self.pred_horizon))
                        for i in range(self.pred_horizon):
                            gear_choice_binary[int(gear_choice_explicit[i])-1, i] = 1

                        if not np.all(np.sum(gear_choice_binary, axis=0) == 1):
                            raise ValueError("gear binaries not correctly set!") 
                        o = 5
                        for i in range(len(Vehicle.b)):
                            for k in range(mpc_agent.mpc.N):
                                mpc_agent.mpc.sigma[i, 0, k].ub = gear_choice_binary[i, k] 
                                mpc_agent.mpc.sigma[i, 0, k].lb = gear_choice_binary[i, k]
                            # for k in range(o, mpc_agent.mpc.N):
                            #     mpc_agent.mpc.sigma[i, 0, k].ub = 1
                            #     mpc_agent.mpc.sigma[i, 0, k].lb = 0
                            
                        u, info = mpc_agent.get_control(state, try_again_if_infeasible=False)     

                        if info["cost"] == float('inf'):
                            j = vehicle[0].get_gear_from_velocity(state[1].item())
                
                            #assume all time step have the same gear choice
                            gear_choice_init_explicit = np.ones((self.pred_horizon, 1)) * j
                            gear_choice_init_binary = np.zeros((self.n_gears, self.pred_horizon))
                            #set value for gear choice binary
                            for i in range(self.pred_horizon):
                                gear_choice_init_binary[int(gear_choice_init_explicit[i])-1, i] = 1

                            for i in range(len(Vehicle.b)):
                                for k in range(mpc_agent.mpc.N):
                                    mpc_agent.mpc.sigma[i, 0, k].ub = gear_choice_init_binary[i, k]
                                    mpc_agent.mpc.sigma[i, 0, k].lb = gear_choice_init_binary[i, k]

                            u, info = mpc_agent.get_control(state, try_again_if_infeasible=False) 

                            if info['cost'] == float('inf'):
                                raise RuntimeError("Backup gear solution was still infeasible, reconsider theory. Oh no.")    

                    infeasibility.append(count)
                                   
                    pred_u_next = info['u']
                    pred_x_next = info['x']
                    # transform the predicition to tensor
                    pred_u_next = torch.tensor(info['u'].T, dtype=torch.float32, device=self.device).unsqueeze(0)
                    pred_x_next = torch.tensor(info['x'][:,:-1].T, dtype=torch.float32, device=self.device).unsqueeze(0)
                    pred_vel_next = pred_x_next[[0], :, 1]
                    pred_pos_next = pred_x_next[[0], :, 0]

                    # Get the reference trajectory
                    ref_pos_next = torch.tensor(mpc_agent.leader_x[0, timestep : timestep + mpc_agent.N], dtype=torch.float32, device=self.device).unsqueeze(0)

                    # Calculate the relative position
                    rel_pos_next = pred_pos_next - ref_pos_next

                    # Normalize the velocity prediction
                    pred_vel_next_norm = (pred_vel_next - env.platoon.vehicles[0].vl[0]) / (env.platoon.vehicles[0].vh[-1] - env.platoon.vehicles[0].vl[0])

                    next_pred_states = torch.cat((rel_pos_next.unsqueeze(-1), pred_vel_next_norm.unsqueeze(-1), pred_u_next), 2)

                    pred_states = next_pred_states
                    last_gear_choice_explicit = gear_choice_explicit


                else:
                    if counter > actions.shape[1]:
                        raise RuntimeError(
                            f"Open loop actions of length {actions.shape[1]} where not enough for episode."
                        )
                    action = actions[:, [counter]]
                    counter += 1

                state, r, truncated, terminated, rewards = env.step(u)
                mpc_agent.on_env_step(env, episode, timestep)

                returns[episode] += r
                fuel_cost = rewards['cost_fuel'] #only care about fuel
                tracking_cost = rewards['cost_tracking']
  
                total_reward += fuel_cost + penalty + tracking_cost
                
                total_tracking_cost += tracking_cost
                total_fuel_cost += fuel_cost
                timestep += 1
                mpc_agent.on_timestep_end(env, episode, timestep)
                
            performance += 0.0025 * total_tracking_cost + total_fuel_cost
            end_time = time.time()
            total_runtime += end_time - start_time
            print(f"Episode {episode + 1}, performance: {performance}, total cost: {total_reward}, runtime: {total_runtime}, gear violations: {gear_violation_count}, infeasibilities: {infeasibility_count}")
            mpc_agent.on_episode_end(env, episode, returns[episode])

            if num_episodes > 1:
                wandb.log({"total_cost": total_reward,
                            "gear_violations": gear_violation_count, 
                            "infeasibilities": infeasibility_count})
            
            self.episode_rewards.append(total_reward)
            self.gear_violations.append(gear_violation_count)
            self.infeasibilities.append(infeasibility_count)
            self.tracking_costs.append(total_tracking_cost)
            self.fuel_costs.append(total_fuel_cost)
            self.performance.append(performance)
            self.runtimes.append(total_runtime)


            if num_episodes > 1 and episode == num_episodes - 1:
                self.plot_rewards()
                self.plot_gear_violation()
                self.plot_infeasibility()


            if plot and num_episodes == 1:
                if len(env.observations) > 0:
                    X = env.observations[episode].squeeze()
                    U = env.actions[episode].squeeze()
                    R = env.rewards[episode]  # grabs only total cost
                else:
                    X = np.squeeze(env.ep_observations)
                    U = np.squeeze(env.ep_actions)
                    R = np.squeeze(env.ep_rewards)

                # now grab individual costs
                r_tracking = env.unwrapped.cost_tracking_list
                r_fuel = env.unwrapped.cost_fuel_list
                acc = env.unwrapped.acc_list
                r_tracking = np.array(r_tracking).squeeze()
                r_fuel = np.array(r_fuel).squeeze()
                acc = np.array(acc).squeeze()

                plot_fleet(
                    1,
                    X,
                    U,
                    R,
                    r_tracking,
                    r_fuel,
                    mpc_agent.leader_x,
                    violations=env.unwrapped.viol_counter[0],
                    )
                
                # plot the gear violations and infeasibilities
                plt.figure()
                plt.plot(gear_violation)
                plt.xlabel('Time step')
                plt.ylabel('Gear Violations')

                plt.figure()
                plt.plot(infeasibility)
                plt.xlabel('Time step')
                plt.ylabel('Infeasibilities')

                plt.show()    
            
                # dump the data
                evaluate_data = {}
                evaluate_data['states'] = X
                evaluate_data['actions'] = U
                evaluate_data['rewards'] = R
                evaluate_data['episode_rewards'] = self.episode_rewards
                evaluate_data['gear_violations'] = self.gear_violations
                evaluate_data['infeasibilities'] = self.infeasibilities
                evaluate_data['tracking_costs'] = r_tracking
                evaluate_data['fuel_costs'] = r_fuel

                if save:
                    with open('evaluate_data.pkl', 'wb') as f:
                        pickle.dump(evaluate_data, f)
            
        mpc_agent.on_validation_end(env, returns)

        print("Complete")
        print(f"Average performance: {np.mean(self.performance)}")
        print(f"Average total cost: {np.mean(self.episode_rewards)}")
        print(f"Average tracking costs: {np.mean(self.tracking_costs)}")
        print(f"Average fuel costs: {np.mean(self.fuel_costs)}")
        print(f"Average gear violations: {np.mean(self.gear_violations)}")
        print(f"Average infeasibilities: {np.mean(self.infeasibilities)}")
        print(f"Average runtime: {np.mean(self.runtimes)}")

 
        # env.render()
        env.close()
        # plt.ioff()
        plt.show()
    
        return returns

    
# Training agent
def simulate(
    sim: Sim,
    save: bool = False,
    plot: bool = True,
    thread_limit: int | None = None,
    leader_index=0,
    mode: str = "train",
):
    n = sim.n  # num cars
    N = sim.N_dqn  # controller horizon
    ep_len = sim.ep_len  # length of episode (sim len)
    ts = Params.ts
    masses = sim.masses

    spacing_policy = sim.spacing_policy
    leader_trajectory = sim.leader_trajectory
    leader_x = leader_trajectory.get_leader_trajectory()
    # vehicles
    platoon = Platoon(n, vehicle_type=sim.vehicle_model_type, masses=masses)
    systems = platoon.get_vehicle_system_dicts(ts)

    # env
    env = MonitorEpisodes(
        TimeLimit(
            PlatoonEnv(
                n=n,
                platoon=platoon,
                leader_trajectory=leader_trajectory,
                spacing_policy=spacing_policy,
                start_from_platoon=sim.start_from_platoon,
                real_vehicle_as_reference=sim.real_vehicle_as_reference,
                ep_len=sim.ep_len,
                leader_index=leader_index,
                quadratic_cost=sim.quadratic_cost,
                fuel_penalize=sim.fuel_penalize,
            ),
            max_episode_steps=ep_len,
        )
    )
    rlagent = DqnAgent(128, 0.9, 0.999, 0, 980000, 0.001, 0.001, "cuda")

    mpc = MpcGearCent(
        n,
        N,
        systems,
        spacing_policy=spacing_policy,
        leader_index=leader_index,
        thread_limit=thread_limit,
        real_vehicle_as_reference=sim.real_vehicle_as_reference,
        quadratic_cost=sim.quadratic_cost,
    )


    mpcagent = TrackingCentralizedAgent(mpc, ep_len, N, leader_x)

    # train or evaluate
    if mode == "train":
        rlagent.train(env, mpcagent, 70000, seed = None, save = True)
    elif mode == "evaluate":
        rlagent.load_model('trained_dqn_agent_full_complexity_fuel_param_1_hidden_size_64.pth')
        rlagent.evaluate(env, mpcagent, num_episodes = 100, seed = Sim.seed, save = save, plot = plot)
    
if __name__ == "__main__": 
    simulate(Sim(), save=True, plot = True, leader_index=0, mode="evaluate")
    