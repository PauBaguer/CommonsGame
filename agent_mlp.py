import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

import gym
from utils import plot_learning_curve



class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, batch_size, out_channels=6, kernel_size=3, stride=1):
        super(DeepQNetwork, self).__init__()
        cpus = int(os.cpu_count() / 2)
        T.set_num_threads(cpus)

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self. n_actions = n_actions
        self.batch_size = batch_size
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        fc1_input_dims = np.prod(input_dims)
        # todo https: // pytorch.org / docs / stable / generated / torch.nn.LayerNorm.html
        self.fc1 = nn.Linear(fc1_input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.to(self.device)

        print(f"Using device: {self.device}, n_procs: {T.get_num_threads()}")

    def forward(self, state):

        x = torch.flatten(state, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions

class Agent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size=100_000, eps_end=0.05, eps_dec=1e-4): # more decay
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0

        self.Q_eval = DeepQNetwork(lr, n_actions=n_actions, input_dims=input_dims,
                                   fc1_dims=32, fc2_dims=32, batch_size=batch_size)
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.int32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.int32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = terminal

        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor(np.array([observation], dtype=np.float32)).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        # Play episodes beginning until memory full.
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch], dtype=torch.float32).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch], dtype=torch.float32).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch) # This to target_network.
        # q_next[terminal_batch] = 0.0

        q_target = reward_batch + (1 - terminal_batch.to(torch.float32)) * self.gamma * T.argmax(q_next, dim=1)[0] # < feo

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        # limitar els gradients perque no siguin grans. -> Gradient clipping if grad > X, limit to threshold.
        #T.nn.utils.clip_grad_value_(self.Q_eval.parameters(), 100)
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min


def test():
    pass
if __name__=="__main__":
    test()