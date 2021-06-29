import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random, time
import numpy as np

class Qnet(nn.Module):
    def __init__(self, learning_rate, gamma):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 3)

        self.gamma = gamma
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.optimization_step = 0

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 2)
        else:
            return out.argmax().item()

    def train_net(self, q_target, memory, batch_size):
        loss_lst = []
        t1 = time.time()

        for i in range(20):
            s, a, r, s_prime, done_mask = memory.sample(batch_size)
            q_out = self.forward(s)
            q_a = q_out.gather(2, a.unsqueeze(1))
            max_q_prime = q_target(s_prime).max(2)[0].unsqueeze(1)
            target = r.unsqueeze(1) + self.gamma * max_q_prime * done_mask.unsqueeze(1)
            loss = F.smooth_l1_loss(q_a, target).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.optimization_step += 1
            loss_lst.append(loss.item())

        train_t = time.time() - t1

        return np.mean(loss_lst), train_t


class QnetConv(nn.Module):
    def __init__(self, learning_rate, gamma):
        super(QnetConv, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, 5)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 32, 5)
        self.pool2 = nn.MaxPool1d(2)

        self.fc1 = nn.Linear(22*32, 256)
        self.fc2 = nn.Linear(256, 3)

        self.gamma = gamma
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.optimization_step = 0

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        x = x.reshape(-1, 22*32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 2)
        else:
            return out.argmax().item()

    def train_net(self, q_target, memory, batch_size):
        loss_lst = []
        t1 = time.time()

        for i in range(20):
            s, a, r, s_prime, done_mask = memory.sample(batch_size)
            q_out = self.forward(s)
            q_a = q_out.gather(1, a)
            max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
            target = r + self.gamma * max_q_prime * done_mask
            loss = F.smooth_l1_loss(q_a, target).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.optimization_step += 1
            loss_lst.append(loss.item())

        train_t = time.time() - t1
        return np.mean(loss_lst),  train_t