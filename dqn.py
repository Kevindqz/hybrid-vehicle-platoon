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
from mpcs.rl_mpc import RlMpcAgent
from gymnasium import Env

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

    def __init__(self, input_size, hidden_size, num_actions = 3, num_layers=1):
        super(DRQN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_actions)

    def forward(self, x):
        # x: (batch_size, seq_length, input_size)
        lstm_out, _ = self.rnn(x)  # lstm_out: (batch_size, seq_length, hidden_size)
        # Apply the fully connected layer to each time step
        q_values = self.fc(lstm_out)  # q_values: (batch_size, seq_length, num_actions)
        return q_values

class DqnAgent():
    """A class that set up RL training scheme with DQN Algorithm and interfaces with Mpc."""
    def __init__(self, batch_size, gamma, eps_start, eps_end, eps_decay, target_update, learning_rate, device)-> None:
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.EPS_START = eps_start
        self.EPS_END = eps_end
        self.EPS_DECAY = eps_decay
        self.TARGET_UPDATE = target_update
        self.LEARNING_RATE = learning_rate
        self.device = device

        self.n_states = 4 # 3 states: position, velocity, gear, 1 action
        self.n_hidden = 128
        self.n_actions = 6

        self.policy_net = DRQN(self.n_states, self.n_hidden).to(self.device)
        self.target_net = DRQN(self.n_states, self.n_hidden).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.LEARNING_RATE, amsgrad=True)
        self.memory = ReplayMemory(10000)
        self.steps_done = 0
        self.episode_durations = []

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold: # Should I do this for every time step?
            with torch.no_grad():
                actions = self.policy_net(state).max(2)[1]
                return actions
        else:  #this random action does not make sense
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def plot_durations(self, show_result=False):
        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        # if is_ipython:
        #     if not show_result:
        #         display.display(plt.gcf())
        #         display.clear_output(wait=True)
        #     else:
        #         display.display(plt.gcf())

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
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def train(self, env: Env, mpc_agent: RlMpcAgent, num_episodes: int):
        """
        This method trains the DQN agent with the given environment and Mpc agent.
        """
        if torch.cuda.is_available():
            num_episodes = 600
        else:
            num_episodes = 50

        for i_episode in range(num_episodes):
            # Initialize the environment and get its state
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            # Initialize the Mpc agent and get its state，need to be rewritten
            
            state_rl = ... 
            state_mpc = ...
            pred_states = ...

            for t in count():
                action = self.select_action(pred_states) # state_seq: (seq_length, input_size)

                gear_choice = ... # Transform action to gear choice
                
                #Interface with Mpc agent
                mpc_agent.pass_gear_choice(gear_choice)
                u, info = mpc_agent.get_control(state) # check raise 
                u_pred = info['u']
                x_pred = info['x']
                
                observation, _ , terminated, truncated, rewards = env.step(u)
               
                fuel_cost = torch.tensor([rewards['fuel']], device=self.device) #only care about fuel
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    u_next, info_next = mpc_agent.get_control(observation)
                    u_pred_next = info_next['u']
                    x_pred_next = info_next['x']
                    # transform the predicition to tensor
                    u_pred_next = torch.tensor(u_pred_next, dtype=torch.float32, device=self.device).unsqueeze(0)
                    x_pred_next = torch.tensor(x_pred_next, dtype=torch.float32, device=self.device).unsqueeze(0)
                    # concatenate the state and action
                    next_pred_states = torch.cat((x_pred_next, u_pred_next), 1)
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

                # Store the transition in memory
                self.memory.push(pred_states, action, next_pred_states, fuel_cost) 

                # Move to the next state
                state = next_state
                pred_states = next_pred_states
                
                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.TARGET_UPDATE + target_net_state_dict[key]*(1-self.TARGET_UPDATE)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    self.episode_durations.append(t + 1)
                    self.plot_durations()
                    break
            
        print('Complete')
        env.render()
        env.close()
        plt.ioff()
        plt.show()

    
# Training the DQN agent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dqn_agent = DqnAgent(128, 0.999, 0.9, 0.05, 200, 0.001, device)
