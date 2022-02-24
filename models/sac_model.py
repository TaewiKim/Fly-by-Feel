import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np


class PolicyNet(nn.Module):
    def __init__(self, learning_rate, init_alpha, lr_alpha, target_entropy):
        super(PolicyNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 32, 5)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 32, 5)
        self.pool2 = nn.MaxPool1d(2)

        self.fc1 = nn.Linear(61 * 32, 256)
        self.fc_mu = nn.Linear(256, 1)
        self.fc_std  = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.log_alpha = torch.tensor(np.log(init_alpha))
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)

        self.optimization_step = 0
        self.target_entropy = target_entropy

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        x = x.reshape(-1, 61 * 32)
        x = F.relu(self.fc1(x))

        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = Normal(mu, std)
        action = dist.rsample()

        log_prob = dist.log_prob(action)
        real_action = torch.tanh(action)
        real_log_prob = log_prob - torch.log(1-torch.tanh(action).pow(2) + 1e-7)

        # print("a", real_action)
        # print("log prob", real_log_prob)
        return real_action, real_log_prob

    def train_net(self, q1, q2, mini_batch):
        s, _, _, _, _ = mini_batch
        a, log_prob = self.forward(s)
        entropy = -self.log_alpha.exp() * log_prob

        q1_val, q2_val = q1(s,a), q2(s,a)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]

        loss = -min_q - entropy # for gradient ascent
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
        self.optimization_step +=1

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

class QNet(nn.Module):
    def __init__(self, learning_rate, tau):
        super(QNet, self).__init__()
        self.tau = tau
        self.conv1 = nn.Conv1d(3, 32, 5)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 32, 5)
        self.pool2 = nn.MaxPool1d(2)

        self.fc1 = nn.Linear(61 * 32, 128)
        self.fc_a = nn.Linear(1, 64)
        self.fc_a2 = nn.Linear(64, 64)
        self.fc_cat1 = nn.Linear(128+64, 128)
        self.fc_cat2 = nn.Linear(128, 32)
        self.fc_out = nn.Linear(32, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x, a):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        x = x.reshape(-1, 61 * 32)
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc_a(a))
        h2 = F.relu(self.fc_a2(h2))
        cat = torch.cat([h1,h2], dim=1)
        q = F.relu(self.fc_cat1(cat))
        q = F.relu(self.fc_cat2(q))
        q = self.fc_out(q)
        return q

    def train_net(self, target, mini_batch):
        s, a, r, s_prime, done = mini_batch
        loss = F.smooth_l1_loss(self.forward(s, a) , target).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def soft_update(self, net_target):
        for param_target, param in zip(net_target.parameters(), self.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

def calc_target(pi, q1, q2, mini_batch, gamma):
    s, a, r, s_prime, done = mini_batch

    with torch.no_grad():
        a_prime, log_prob= pi(s_prime)
        entropy = -pi.log_alpha.exp() * log_prob
        entropy = entropy.mean(1, keepdim=True)
        q1_val, q2_val = q1(s_prime,a_prime), q2(s_prime,a_prime)

        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]
        target = r + gamma * done * (min_q + entropy)

    return target.float()