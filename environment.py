import random

import numpy as np
from utils.serialChannel import serialPlot
import copy
import math
from scipy.stats import multivariate_normal

from collections import deque


class Environment:
    def __init__(self, config, dw_thread, serial_channel: serialPlot, streamingClient):
        self.config = config
        self.target_position = config["target_position"]
        self.dw_thread = dw_thread
        self.serial_channel = serial_channel
        self.streamingClient = streamingClient
        self.step_count = 0
        self.Px_sum = 0
        self.Py_sum = 0
        self.Pz_sum = 0
        self.Drone_position = 0
        self.human_action = 0

        self.n_state = 256

        self.action_right_queue = deque(maxlen=self.n_state)
        self.action_left_queue = deque(maxlen=self.n_state)

    def init_action_queue(self):
        for i in range(self.n_state):
            self.action_right_queue.append(0.0)
            self.action_left_queue.append(0.0)


    def reset(self):
        self.stop_drone()
        self.step_count = 0
        self.max_Px, self.min_Px = -100000.0, 1000000.0
        self.max_Py, self.min_Py = -100000.0, 1000000.0
        self.max_Pz, self.min_Pz = -100000.0, 1000000.0
        self.max_s, self.min_s = -100000.0, 1000000.0
        self.init_action_queue()


    def get_current_state(self):
        input_state = self.dw_thread.state
        # print(input_state)
        self.human_action = self.dw_thread.human_action

        reward, done, Drone_position = self.calc_reward_done()
        self.max_Px = max(self.max_Px, Drone_position[0])
        self.min_Px = min(self.min_Px, Drone_position[0])
        self.Px_sum += Drone_position[0]
        self.max_Py = max(self.max_Py, Drone_position[1])
        self.min_Py = min(self.min_Py, Drone_position[1])
        self.Py_sum += Drone_position[1]
        self.max_Pz = max(self.max_Pz, Drone_position[2])
        self.min_Pz = min(self.min_Pz, Drone_position[2])
        self.Pz_sum += Drone_position[2]

        if done:
            self.stop_drone()

        self.max_s = max(self.max_s, np.max(input_state))
        self.min_s = min(self.min_s, np.min(input_state))

        #fin_state = np.concatenate((np.array(input_state), [action_right, action_left]), axis=0)

        #return fin_state, reward, done, Drone_position
        return np.array(input_state), reward, done, Drone_position


    def step(self, actions):
        a_thrust = actions[0]   # a_thrust : -1 ~ 1
        a_direction = actions[1]

        action_tail = (a_direction) * 120 # action : real number between -150~150, motor power

        action_front = ((a_thrust + 1) / 2.0) * 200 + 50 # action : real number between 0 ~ 250, motor power
        if action_front < 55:
            action_front = 0

        action_str = "T" + str(int(action_front)) + "%" + "D" + str(int(action_tail)) + "%"

        self.step_count += 1

        self.serial_channel.serialConnection.write(action_str.encode())
        for i in range(int(1280 * self.config["decision_period"])):
            self.action_right_queue.append(a_thrust)
            self.action_left_queue.append(a_direction)


    def calc_reward_done(self):
        done = False
        Drone_position = np.array(self.streamingClient.pos)*1000
        # Drone_rotation = self.streamingClient.rot
        # print(Drone_position)

        mu = self.target_position
        cov = [[500000, 0, 0], [0, 500000, 0], [0, 0, 100000000]]
        rv = multivariate_normal(mu, cov)
        reward = rv.pdf([Drone_position[0], Drone_position[1], Drone_position[2]])*10**10*3

        if Drone_position[2] < -2500:
            reward = 0

        if Drone_position[2] > 4800:
            done = True
            print('Z > 4800')

        if abs(Drone_position[0]) > 1800:
            done = True
            print('X > 2000')


        if self.step_count >= self.config["max_episode_len"]:
            done = True

        return reward, done, Drone_position


    def stop_drone(self):
        action_str = "T" + str(0) + "%" + "D" + str(0) + "%"
        self.serial_channel.serialConnection.write(action_str.encode())


    @classmethod
    def close(cls):
        pass


class DummyEnv:
    def __init__(self):
        self.step_count = 0

    def reset(self):
        return np.random.rand(32)

    def step(self, action):
        self.step_count += 1

        return np.random.rand(32), +0.01, False


