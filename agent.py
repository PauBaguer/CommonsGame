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
        cpus=os.cpu_count()
        T.set_num_threads(cpus)
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self. n_actions = n_actions
        self.batch_size = batch_size
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride
        )
        out_size_height = np.floor(( input_dims[0] - kernel_size ) / stride) + 1
        out_size_width = np.floor((input_dims[1] - kernel_size) / stride) + 1
        conf_out_dims = int(out_size_width * out_size_height * out_channels)

        self.fc1 = nn.Linear(conf_out_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.to(self.device)

        print(f"Using device: {self.device}, n_procs: {T.get_num_threads()}")

    def forward(self, state):
        input_tensor = state.permute(0, 3, 1, 2)
        x = self.conv(input_tensor)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions

class Agent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size=100_000, eps_end=0.05, eps_dec=1e-5):
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
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min



def test_agent():
    env = gym.make('LunarLander-v2',
                   enable_wind=False,
                   wind_power=0.0,
                   turbulence_power=0.0)

    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4,
                  eps_end=0.01, input_dims=[8], lr=0.003)
    scores, eps_history = [], []
    n_games = 500

    for i in range(n_games):
        score = 0
        done = False
        observation, _ = env.reset()

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, truncated,  info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)

            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        print('Episode {} Average Score: {:.2f} Epsilon {:.2f}'.format(i, avg_score, agent.epsilon))

    x = [i+1 for i in range(n_games)]
    filename = 'lunar_lander_test.png'
    plot_learning_curve(x, scores, eps_history, filename)



if __name__ == '__main__':
    test_agent()

