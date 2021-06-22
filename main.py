import random
import numpy as np
from utils.dwserver import *
from utils.dwclient import *
from utils.serialChannel import *

import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

DATA_TYPES = ['b', 'B', 'h', 'H', 'i', 'f', 'q', 'd']
DATA_SIZE = [1, 1, 2, 2, 4, 4, 8, 8]


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(1, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 2)
        else:
            return out.argmax().item()


class Environment():
    def __init__(self, dw_thread, serialChannel):
        self.dw_thread = dw_thread
        self.serial_channel = serialChannel

    def reset(self):
        PWM = 0
        MtrSpd = 'S' + str(PWM) + '%'  # '%' is our ending marker
        self.serial_channel.serialConnection.write(MtrSpd.encode())

        return np.array(self.dw_thread.channel_data)

    def step(self, action):
        if action == 0:
            PWM = 0
            MtrSpd = 'S' + str(PWM) + '%'  # '%' is our ending marker
            self.serial_channel.serialConnection.write(MtrSpd.encode())
            # stay

        elif action == 1:
            PWM = 150
            MtrSpd = 'S' + str(PWM) + '%'  # '%' is our ending marker
            self.serial_channel.serialConnection.write(MtrSpd.encode())
            # go slow

        elif action == 2:
            PWM = 300
            MtrSpd = 'S' + str(PWM) + '%'  # '%' is our ending marker
            self.serial_channel.serialConnection.write(MtrSpd.encode())
            # go fast

        angle = self.serial_channel.getSerialData()
        next_state = np.array(self.dw_thread.channel_data)
        print("state", next_state)

        reward = 0
        done = False

        if angle > 0:
            reward = angle
        if angle < -10:
            self.warning += 1
            done = False
            if self.warning >= 500:
                done = True
                PWM = 0
                MtrSpd = 'S' + str(PWM) + '%'  # '%' is our ending marker
                self.serial_channel.serialConnection.write(MtrSpd.encode())
                reward = -3000
                print('low angle')

        return next_state, reward, done

def main():
    # Teensy Transaction
    portName = 'COM3'
    baudRate = 19200
    maxPlotLength = 100  # number of points in x-axis of real time plot
    dataNumBytes = 4  # number of bytes of 1 data point
    numPlots = 1  # number of plots in 1 graph
    s = serialPlot(portName, baudRate, maxPlotLength, dataNumBytes,
                   numPlots)  # initializes all required variables
    s.readSerialStart()  # starts background thread

    my_thread, tn, ready = get_dewe_thread()
    my_thread.start()
    tn.write(b"STARTTRANSFER 8001\r\n")
    print(tn.read_some())
    ready.wait()
    tn.write(b"STARTACQ\r\n")

    env = Environment(my_thread, s)

    agent = Qnet()

    # state_size = 2
    # action_size = 3
    # agent = DQNAgent(state_size, action_size)

    scores, episodes = [], []
    leftWing, rightWing = [], []
    time.sleep(1)

    state = env.reset()

    for i in range(100):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        print(state_tensor.size())
        action = agent.sample_action(state_tensor, 0.5)
        print(action)
        next_state, reward, done = env.step(action)
        time.sleep(1)
        next_state, reward, done = env.step(0)
        time.sleep(1)
        state = next_state

    # while True:
    #     # agent.step(my_thread, s, 1)
    #     # print(s.getSerialData())
    #     # print(my_thread.channel_data)
    #     time.sleep(0.01)
    #     agent.step(my_thread, s, 0)
    #     print(s.getSerialData())
    #     print(my_thread.channel_data)
    #     time.sleep(0.1)


    # num_episode = 2000
    #
    # for e in range(num_episode):
    #     time.sleep(2)
    #     s.warning = 0
    #     done = False
    #     score = 0
    #     score_avg = 0
    #     count = 0
    #     monitoring = []
    #     # thread = Thread(target=s.getSerialData)
    #     # thread.start()
    #     state = np.array([my_thread.channel_data, my_thread.channel_data])
    #     state = np.reshape(state, [1, -1, state_size, 1])
    #
    #     while not done:
    #         # 현재 상태로 행동을 선택
    #         action = agent.get_action(state)
    #         # 선택한 행동으로 환경에서 한 타임스텝 진행
    #         next_state_part, reward, done = agent.step(my_thread, s, action)
    #         next_state = np.array([next_state_part, next_state_part])
    #         # monitoring.append(next_state)
    #         next_state = np.reshape(next_state, [1, -1, state_size, 1])
    #
    #         # 타임스텝마다 보상 0.1, 에피소드가 중간에 끝나면 -100 보상
    #         score += reward
    #
    #         # 리플레이 메모리에 샘플 <s,a,r,s'> 저장
    #         agent.append_sample(state, action, reward, next_state, done)
    #         # print(state, ',', action, ',', reward, ',', next_state, ',', done)
    #         # 메모리에 데이터 1000개 이상 누적되면 학습 시작
    #
    #         if len(agent.memory) >= agent.train_start:
    #             agent.train_model()
    #
    #         # print(s.angle)
    #         state = next_state
    #         count += 1
    #
    #         if count >= 1000:
    #             PWM = 0
    #             MtrSpd = 'S' + str(PWM) + '%'  # '%' is our ending marker
    #             s.serialConnection.write(MtrSpd.encode())
    #             done = True
    #
    #         if done:
    #             # 각 에피소드마다 타깃 모델을 모델의 가중치로 업데이트
    #             agent.update_target_model()
    #             # 에피소드마다 학습 결과 출력
    #             score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
    #             print(
    #                 "episode: {:3d} | score: {:3.2f} | memory length: {:4d} | epsilon: {:.4f} | trial: {:3d}".format(e, score,
    #                                                                                                       len(
    #                                                                                                           agent.memory),
    #                                                                                                       agent.epsilon, count))
    #
    #             # 이동 평균이 400 이상 때 종료
    #             if score_avg > 50000:
    #                 agent.model.save_weights("./data", save_format="tf")
    #                 sys.exit()
    #         s.isRun = False
    # s.close()

if __name__ == "__main__":
    main()