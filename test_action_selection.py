import random
import torch
import math
import torch.nn as nn
class DRQN(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions=3, num_layers=1):
        super(DRQN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_actions)

    def forward(self, x, hidden=None):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden
    
class Agent:
    def __init__(self, policy_net, n_actions, device):
        self.policy_net = policy_net
        self.n_actions = n_actions
        self.device = device
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200
        self.steps_done = 0

    def select_action(self, state, mode='train'):
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        sample = random.random()
        actions = torch.zeros((1, state.size(1)), device=self.device, dtype=torch.int64)
        current_gear = state[0, 0, 3]
        print("Current gear:", current_gear)
        hidden = None

        if sample > 0:
            with torch.no_grad():
                for i in range(state.size(1)):
                    q_values, hidden = self.policy_net(state[:, i, :].unsqueeze(1), hidden)
                    if current_gear == 6:  # if current gear is 6, upshift is not allowed
                        q_values[0, 0, 2] = -float('inf')
                    elif current_gear == 1:  # if current gear is 1, downshift is not allowed
                        q_values[0, 0, 0] = -float('inf')
                    actions[0, i] = q_values.argmax(2)
                    current_gear = current_gear + actions[0, i].item() - 1
        else:
            for i in range(state.size(1)):
                if current_gear == 6:
                    actions[0, i] = random.choice([0, 1])
                elif current_gear == 1:
                    actions[0, i] = random.choice([1, 2])
                else:
                    actions[0, i] = random.randrange(0, self.n_actions)
                current_gear = current_gear + actions[0, i].item() - 1

        return actions

    def select_action_whole_sequence(self, state):
        with torch.no_grad():
            q_values, _ = self.policy_net(state, None)
            actions = q_values.argmax(2)
        return actions

# 示例调用
input_size = 10
hidden_size = 20
num_actions = 3
policy_net = DRQN(input_size, hidden_size, num_actions)
agent = Agent(policy_net, num_actions, 'cpu')

# 假设 state 是一个形状为 (batch_size, seq_length, input_size) 的张量
state = torch.randn(1, 5, input_size)

# 逐个时间步输入
actions_step_by_step = agent.select_action(state, mode='train')
print("Actions (step by step):", actions_step_by_step)

# 一次性输入整个序列
actions_whole_sequence = agent.select_action_whole_sequence(state)
print("Actions (whole sequence):", actions_whole_sequence)

# 示例调用
input_size = 10
hidden_size = 20
num_actions = 3
policy_net = DRQN(input_size, hidden_size, num_actions)
agent = Agent(policy_net, num_actions, 'cpu')

# 假设 state 是一个形状为 (batch_size, seq_length, input_size) 的张量
state = torch.randn(1, 5, input_size)
# agent.pred_horizon = state.size(1)

# 选择动作
actions = agent.select_action(state, mode='train')
print("Actions:", actions)