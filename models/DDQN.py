import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Qvalue(nn.Module):
    def __init__(self, input, output):
      super(Qvalue, self).__init__()
      self.fc1 = nn.Linear(input, 64)
      self.fc2 = nn.Linear(64, 64)
      self.Q = nn.Linear(64, output)

    def forward(self, s):
      x = self.fc1(s)
      x = F.relu(x)
      x = self.fc2(x)
      x = F.relu(x)
      output = self.Q(x)

      return output

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.orthogonal_(m.weight)

def copy_target(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def soft_update(target_model, local_model, tau):
      for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
          target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, terminated, truncated):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, terminated, truncated)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        state, action, reward, next_state, terminated, truncated = map(np.stack, zip(*batch))
        return state, action, reward, next_state, terminated, truncated

    def __len__(self):
        return len(self.buffer)

class Agent:
    def __init__(self):

        Q = Qvalue(env.observation_space.shape[0], env.action_space.n).to(device)
        Q_target = Qvalue(env.observation_space.shape[0], env.action_space.n).to(device)

        Q.apply(init_weights)  #

        copy_target(Q_target, Q)  #

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(Q.parameters(), lr=5e-4)

        ER = ReplayMemory(int(1e6), 1)

        # Parameters of learning
        gamma = 0.99
        epsilon = 1

        # NN parameters
        batch_size = 64

    def e_greedy_policy(self, Qs):
        return env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(Qs)

    def rollout(self, niter):
        G = 0
        for i in range(niter):
            state, _ = env.reset()
            for _ in range(1000):
                Qs = Q(torch.tensor(state).to(device)).detach().cpu().numpy()
                action = np.argmax(Qs)
                next_state, reward, terminated, truncated, info = env.step(action)
                G += reward
                if truncated or terminated: break
                state = next_state
        return G / niter