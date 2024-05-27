import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


class Qvalue(nn.Module):
    def __init__(self, input_dims, n_actions):
      super(Qvalue, self).__init__()

      self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

      self.input_dims = input_dims
      self.n_actions = n_actions

      self.conv1 = nn.Conv2d(in_channels=input_dims[2], out_channels=32, kernel_size=3, stride=3)
      self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=2)
      self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1)
      
      self.fc1 = nn.Linear(64, 64)
      #self.fc2 = nn.Linear(64, 64)
      self.Q = nn.Linear(64, n_actions)

      self.to(self.device)

    def forward(self, s):
        s = s.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(s))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = T.flatten(x, 1)

        x = F.relu(self.fc1(x))

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
    def __init__(self, input_dims, n_actions):


        self.Q = Qvalue(input_dims, n_actions)
        self.Q_target = Qvalue(input_dims, n_actions)

        self.Q.apply(init_weights)  #

        copy_target(self.Q_target, self.Q)  #

        self.action_space = [i for i in range(n_actions)]

        self.criterion = nn.MSELoss()
        self.optimizer = T.optim.Adam(self.Q.parameters(), lr=5e-4)

        self.ER = ReplayMemory(int(1e6), 1)

        # Parameters of learning
        self.gamma = 0.99
        self.epsilon = 1

        # NN parameters
        self.batch_size = 64


    def e_greedy_policy(self, Qs):
        return np.random.choice(self.action_space) if (np.random.random() <= self.epsilon) else np.argmax(Qs)

    def store_transition(self, state, action, reward, next_state, terminated):
        self.ER.push(state, action, reward, next_state, terminated, False)

    # def rollout(self, niter):
    #     G = 0
    #     for i in range(niter):
    #         state, _ = env.reset()
    #         for _ in range(1000):
    #             Qs = self.Q(torch.tensor(state).to(device)).detach().cpu().numpy()
    #             action = np.argmax(Qs)
    #             next_state, reward, terminated, truncated, info = env.step(action)
    #             G += reward
    #             if truncated or terminated: break
    #             state = next_state
    #     return G / niter

    def learn(self):
        # Train
        if self.batch_size <= len(self.ER):
            # Get batch from exprience replay
            state_batch, action_batch, reward_batch, next_state_batch, terminated_batch, truncated_batch = self.ER.sample(
                self.batch_size)
            state_batch = T.FloatTensor(state_batch).to(self.Q.device)
            action_batch = T.Tensor(action_batch).to(dtype=T.long).to(self.Q.device).unsqueeze(1)
            next_state_batch = T.FloatTensor(next_state_batch).to(self.Q.device)
            reward_batch = T.FloatTensor(reward_batch).to(self.Q.device).unsqueeze(1)
            terminated_batch = T.FloatTensor(terminated_batch).to(dtype=T.long).to(self.Q.device).unsqueeze(1)
            #truncated_batch = T.FloatTensor(truncated_batch).to(dtype=T.long).to(self.device).unsqueeze(1)

            # q_targets_next = torch.gather(Q_target(next_state_batch).detach(),1,torch.argmax(Q(next_state_batch).detach(),dim=1,keepdim = True))    # Double Q-learning
            q_targets_next = T.max(self.Q_target(next_state_batch).detach(), dim=1, keepdim=True)[0]  # Standard DQN

            # target = reward_batch+  (1-(truncated_batch + terminated_batch)) *gamma*q_targets_next
            target = reward_batch + (1 - (terminated_batch)) * self.gamma * q_targets_next

            q_expected = T.gather(self.Q(state_batch), 1, action_batch)

            loss = self.criterion(q_expected, target)  # MSE
            self.optimizer.zero_grad()
            loss.backward()
            # In-place gradient clipping
            #T.nn.utils.clip_grad_value_(self.Q.parameters(), 100)
            self.optimizer.step()

            soft_update(self.Q_target, self.Q, 1e-3)
            return loss.detach().cpu().numpy()

    def choose_action(self, state):
        if np.random.random() > self.epsilon:
            state = T.tensor(np.array([state], dtype=np.float32)).to(self.Q.device)
            actions = self.Q.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action


    def epsilon_decay(self):
        self.epsilon = max(0.1, 0.99 * self.epsilon)