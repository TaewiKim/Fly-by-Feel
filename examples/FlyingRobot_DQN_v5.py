import os
from typing import Union
import telnetlib
import sys
from threading import Event
import dwserver
import serial
import sys
import pylab
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform

from threading import Thread
import time
import collections
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
import struct
import copy
import pandas as pd

DATA_TYPES = ['b', 'B', 'h', 'H', 'i', 'f', 'q', 'd']
DATA_SIZE = [1, 1, 2, 2, 4, 4, 8, 8]

class Channel:
    def __init__(self, input, sample_rate):
        self.ch = input[0]
        self.channel_type = input[1]
        self.num = int(input[2])
        self.name = input[3]
        self.desc = input[4]
        self.unit = input[5]
        self.timestamp = []
        self.channel_data = []
        self.number_of_added_samples = 0
        self.sample_rate = sample_rate

        self.async_ch = False
        self.single_value = False
        if input[6] == "Async":
            self.async_ch = True
        elif input[6] == "SingleValue":
            self.single_value = True
        else:
            self.sample_div = int(input[6])
        self.expected_async_rate = float(input[7])
        self.measur_type = int(input[8])
        self.data_type = DATA_TYPES[int(input[9])]
        self.data_type_size = DATA_SIZE[int(input[9])]
        self.buffer_size = int(input[10])
        self.custom_scale = float(input[11])
        self.custom_offset = float(input[12])
        self.scale_raw_data = float(input[13].replace(",", "."))
        self.offset_raw_data = float(input[14])
        self.description = input[15]
        self.settings = input[16]
        self.range_min = float(input[17].replace(",", "."))
        self.range_max = float(input[18].replace(",", "."))
        if input[19] == 'OvlYes':
            self.can_overload = True
        elif input[19] == 'OvlNo':
            self.can_overload = False
        else:
            self.can_overload = False
        self.auto_zero = bool(input[20])
        if int(input[21]) > 0:
            self.discrete_list = [element for element in input[22: 22 + int(21)]]
        else:
            self.discrete_list = []
        # self.current_min = float(input[21 + int(input[21]) + 1].replace(",", "."))
        # self.current_max = float(input[21 + int(input[21]) + 2].replace(",", "."))
        # self.current_avg = float(input[21 + int(input[21]) + 3].replace(",", "."))


def process_listusedchs(input, sample_rate):
    list_of_used_ch = []
    input = input.decode('utf-8').split('\r\n')[1:-2]
    for element in input:
        output_element = element.split('\t')
        list_of_used_ch.append(Channel(output_element, sample_rate))
    return list_of_used_ch


def prepare_channels(selected_channels, input_array):
    result_array = []
    for element in selected_channels:
        result_array.append(input_array[element])
    return result_array


class serialPlot:
    def __init__(self, serialPort, serialBaud, plotLength, dataNumBytes, numPlots):
        self.port = serialPort
        self.baud = serialBaud
        self.plotMaxLength = plotLength
        self.dataNumBytes = dataNumBytes
        self.numPlots = numPlots
        self.rawData = bytearray(numPlots * dataNumBytes)
        self.dataType = None
        if dataNumBytes == 2:
            self.dataType = 'h'  # 2 byte integer
        elif dataNumBytes == 4:
            self.dataType = 'f'  # 4 byte float
        self.angle = 0
        self.isRun = True
        self.isReceiving = False
        self.thread = None
        self.thread2 = None
        self.data = []
        for i in range(numPlots):  # give an array for each type of data and store them in a list
            self.data.append(collections.deque([0] * plotLength, maxlen=plotLength))
        self.plotTimer = 0
        self.previousTimer = 0
        self.csvData = []
        self.buffer = collections.deque(maxlen=100)


        print('Trying to connect to: ' + str(serialPort) + ' at ' + str(serialBaud) + ' BAUD.')
        try:
            self.serialConnection = serial.Serial(serialPort, serialBaud, timeout=4)
            print('Connected to ' + str(serialPort) + ' at ' + str(serialBaud) + ' BAUD.')
        except:
            print("Failed to connect with " + str(serialPort) + ' at ' + str(serialBaud) + ' BAUD.')

    def readSerialStart(self):
        if self.thread == None:
            self.thread = Thread(target=self.backgroundThread)
            self.thread.start()

            # Block till we start receiving values
            while self.isReceiving != True:
                time.sleep(0.1)

    def getSerialData(self):
        for _ in range (100):
            currentTimer = time.perf_counter()
            self.plotTimer = int((currentTimer - self.previousTimer) * 1000)     # the first reading will be erroneous
            self.previousTimer = currentTimer
            privateData = copy.deepcopy(self.rawData[:])    # so that the 3 values in our plots will be synchronized to the same sample time
            data = privateData[0:(self.dataNumBytes)]
            value,  = struct.unpack(self.dataType, data)
            self.data.append(value)    # we get the latest data point and append it to our array
            self.csvData.append([currentTimer, self.data[-1]])
        time.sleep(0.01)

    # def getSerialData(self):
    #     for _ in range (100):
    #         currentTimer = time.perf_counter()
    #         self.plotTimer = int((currentTimer - self.previousTimer) * 1000)     # the first reading will be erroneous
    #         self.previousTimer = currentTimer
    #         privateData = copy.deepcopy(self.rawData[:])    # so that the 3 values in our plots will be synchronized to the same sample time
    #         for i in range(self.numPlots):
    #             data = privateData[(i*self.dataNumBytes):(self.dataNumBytes + i*self.dataNumBytes)]
    #             value,  = struct.unpack(self.dataType, data)
    #             self.data[i].append(value)    # we get the latest data point and append it to our array
    #             self.csvData.append([currentTimer, self.data[0][-1], self.data[1][-1], self.data[2][-1]])
    #     print(self.data[2][-1])
    #     time.sleep(0.01)


    def backgroundThread(self):    # retrieve data
        time.sleep(0.1)  # give some buffer time for retrieving data
        self.serialConnection.reset_input_buffer()
        while (self.isRun):
            self.serialConnection.readinto(self.rawData)
            self.isReceiving = True
            # print(self.rawData)

    def close(self):
        self.isRun = False
        self.thread.join()
        self.serialConnection.close()
        print('Disconnected...')
        df = pd.DataFrame(self.csvData)
        df.to_csv('./data/data.csv')

# 상태 입력, 큐함수 출력인 인공신경망
# DQN 클래스는 tf.keras.Model 클래스의 메소드 상속받음
# super()는 override 방지하기 위함
class DQN(tf.keras.Model):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.fc1 = Conv2D(filters=32, kernel_size=(4, 2), input_shape=(-1, 2, 1))
        self.fc2 = Conv2D(filters=64, kernel_size=(4, 1))
        self.fc3 = Flatten()
        self.fc4 = Dense(128, activation='relu')
        self.fc_out = Dense(action_size, activation='softmax',
                            kernel_initializer=RandomUniform(-1e-3, 1e-3))

    def q_out(self, x):
        y = self.fc1(x)
        y = self.fc2(y)
        y = self.fc3(y)
        y = self.fc4(y)
        q = self.fc_out(y)
        return q

# 카트폴 예제에서의 DQN 에이전트
class DQNAgent():
    def __init__(self, state_size, action_size):

        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size
        self.angle = 0
        self.warning = 0

        # DQN 하이퍼파라미터
        self.discount_factor = 0.9999
        self.learning_rate = 0.0001
        self.epsilon = 1.0
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.05
        self.batch_size = 32
        self.train_start = 5000

        # 리플레이 메모리, 최대 크기 2000
        self.memory = collections.deque(maxlen=10000)

        # 모델과 타깃 모델 생성
        self.model = DQN(action_size)
        self.target_model = DQN(action_size)

        # 타깃 모델 초기화
        self.update_target_model()
        self.optimizer = Adam(lr=self.learning_rate)

    # 타깃 모델을 모델의 가중치로 업데이트
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.q_out(state)
            return np.argmax(q_value[0])

    # motor action
    def step(self, serial, action):
        # SPI 통신으로 motor control

        if action == 0:
            self.PWM = 0
            self.MtrSpd = 'S' + str(self.PWM) + '%'  # '%' is our ending marker
            serial.serialConnection.write(self.MtrSpd.encode())
            # stay

        elif action == 1:
            self.PWM = 100
            self.MtrSpd = 'S' + str(self.PWM) + '%'  # '%' is our ending marker
            serial.serialConnection.write(self.MtrSpd.encode())
            # go slow

        elif action == 2:
            self.PWM = 150
            self.MtrSpd = 'S' + str(self.PWM) + '%'  # '%' is our ending marker
            serial.serialConnection.write(self.MtrSpd.encode())
            # go fast

        thread = Thread(target=serial.getSerialData)
        thread.start()
        self.angle = serial.data
        print(self.angle)

        reward = 0
        done = False

        if self.angle > 0:
            reward += self.angle
        if 0 > self.angle > -5:
            reward += -10
        if self.angle < -10:
            self.warning += 1
            done = False
            if self.warning >= 500:
                done = True
                self.PWM = 0
                self.MtrSpd = 'S' + str(self.PWM) + '%'  # '%' is our ending marker
                serial.serialConnection.write(self.MtrSpd.encode())
                reward = -3000
                print('low angle')

        return next_state, reward, done


    # 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 메모리에서 배치 크기만큼 무작위로 샘플 추출
        mini_batch = random.sample(self.memory, self.batch_size)
        states = np.array([sample[0][0] for sample in mini_batch])
        actions = np.array([sample[1] for sample in mini_batch])
        rewards = np.array([sample[2] for sample in mini_batch])
        next_states = np.array([sample[3][0] for sample in mini_batch])
        dones = np.array([sample[4] for sample in mini_batch])

        # 학습 파라미터
        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            # 현재 상태에 대한 모델의 Q함수
            predicts = self.model.q_out(states)
            one_hot_action = tf.one_hot(actions, self.action_size)
            predicts = tf.reduce_sum(one_hot_action * predicts, axis=1)

            # 다음 상태에 대한 타깃 모델의 Q함수
            target_predicts = self.target_model.q_out(next_states)
            target_predicts = tf.stop_gradient(target_predicts)

            # 벨만 최적 방정식 이용한 업데이트 타깃
            max_q = np.amax(target_predicts, axis=1)
            targets = rewards + (1 - dones) * self.discount_factor * max_q
            loss = tf.reduce_mean(tf.square(targets - predicts))

        # 오류함수를 줄이는 방향으로 모델 업데이트
        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))

    # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))



def main():
    # SETUP MUST BE OPENED IN DEWESOFT
    ready = Event()
    HOST = 'localhost'  # if you want to connect remotly enter IP of pc
    tn = telnetlib.Telnet(HOST, '8999')  # 8999 is standard port number
    tn.read_some()

    tn.write(b"SETMODE 1\r\n")  # we change to control mode
    print(tn.read_some())

    tn.write(b"GETSAMPLERATE\r\n")
    output = tn.read_some().decode('utf-8')

    start = output.find("+OK ") + len("+OK ")
    end = output.find("\r\n")
    sample_rate_str = output[start:end]

    tn.write(b"ISMEASURING\r\n")
    output = tn.read_some().decode('utf-8')
    if output == "+OK Yes\r\n":
        tn.write(b"STOP\r\n")
        tn.read_some()

    tn.write(b"LISTUSEDCHS\r\n")  # here we get list of all used channels
    list_of_used_ch = process_listusedchs(tn.read_until(b'+ETX end list\r\n'), float(sample_rate_str))

    list_of_used_ch = prepare_channels([0], list_of_used_ch)  # filter channels
    tn.write(
        b'/stx preparetransfer\r\nCH 0\r\nCH 1\r\nCH 2\r\n/etx\r\n')  # here we select which channels we want to transfer
    print(tn.read_some())

    my_thread = dwserver.MyThread(ready, list_of_used_ch)
    my_thread.start()
    tn.write(b"STARTTRANSFER 8001\r\n")
    print(tn.read_some())
    ready.wait()
    tn.write(b"STARTACQ\r\n")
    # plot = dwserver.DewePlot(ready, list_of_used_ch)
    # plt.show()
    # my_thread.join()


    portName = 'COM3'
    baudRate = 19200
    maxPlotLength = 100  # number of points in x-axis of real time plot
    dataNumBytes = 4  # number of bytes of 1 data point
    numPlots = 1  # number of plots in 1 graph
    s = serialPlot(portName, baudRate, maxPlotLength, dataNumBytes, numPlots)  # initializes all required variables
    s.readSerialStart()  # starts background thread
    state_size = 1
    action_size = 3
    agent = DQNAgent(state_size, action_size)

    scores, episodes = [], []
    leftWing, rightWing = [], []

    num_episode = 2000


    for e in range(num_episode):
        time.sleep(2)
        s.warning = 0
        done = False
        score = 0
        score_avg = 0
        count = 0
        monitoring = []
        thread = Thread(target=s.getSerialData)
        thread.start()
        state = s.data
        state = np.reshape(state, [1, -1, state_size, 1])

        while not done:
            # 현재 상태로 행동을 선택
            action = agent.get_action(state)
            # 선택한 행동으로 환경에서 한 타임스텝 진행
            next_state, reward, done = agent.step(s, action)
            monitoring.append(next_state)
            next_state = np.reshape(next_state, [1, -1, state_size, 1])

            # 타임스텝마다 보상 0.1, 에피소드가 중간에 끝나면 -100 보상
            score += reward

            # 리플레이 메모리에 샘플 <s,a,r,s'> 저장
            agent.append_sample(state, action, reward, next_state, done)
            # print(state, ',', action, ',', reward, ',', next_state, ',', done)
            # 메모리에 데이터 1000개 이상 누적되면 학습 시작

            if len(agent.memory) >= agent.train_start:
                agent.train_model()

            # print(s.angle)
            state = next_state
            count += 1

            if count >= 1000:
                PWM = 0
                MtrSpd = 'S' + str(PWM) + '%'  # '%' is our ending marker
                s.serialConnection.write(MtrSpd.encode())
                done = True

            if done:
                # 각 에피소드마다 타깃 모델을 모델의 가중치로 업데이트
                agent.update_target_model()
                # 에피소드마다 학습 결과 출력
                score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
                print(
                    "episode: {:3d} | score: {:3.2f} | memory length: {:4d} | epsilon: {:.4f} | trial: {:3d}".format(e, score,
                                                                                                          len(
                                                                                                              agent.memory),
                                                                                                          agent.epsilon, count))

                # 에피소드마다 학습 결과 그래프로 저장
                scores.append(score_avg)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.xlabel("episodes")
                pylab.ylabel("average score")
                pylab.savefig("./data/graph.png")

                # 이동 평균이 400 이상 때 종료
                if score_avg > 50000:
                    agent.model.save_weights("./data", save_format="tf")
                    sys.exit()
            s.isRun = False
    s.close()

if __name__ == "__main__":
    main()