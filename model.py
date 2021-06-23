import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

class Qnet(nn.Module):
    def __init__(self, learning_rate, gamma):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(32, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 3)

        self.gamma = gamma
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

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
        for i in range(100):
            s, a, r, s_prime, done_mask = memory.sample(batch_size)
            q_out = self.forward(s)
            q_a = q_out.gather(2, a.unsqueeze(1))
            max_q_prime = q_target(s_prime).max(2)[0].unsqueeze(1)
            target = r.unsqueeze(1) + self.gamma * max_q_prime * done_mask.unsqueeze(1)
            loss = F.smooth_l1_loss(q_a, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()