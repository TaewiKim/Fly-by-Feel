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
        self.angle_sum = 0
        self.cur_angle = 0
        self.Drone_position = 0
        self.init_angle = 0

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
        self.cur_angle = 0
        self.max_Px, self.min_Px = -100000.0, 1000000.0
        self.max_s, self.min_s = -100000.0, 1000000.0
        self.init_action_queue()


    def get_current_state(self, prev_drone_position):
        input_state = self.dw_thread.state
        action_right = np.array(self.action_right_queue)
        action_left = np.array(self.action_left_queue)

        reward, done, Drone_position = self.calc_reward_done(prev_drone_position)
        self.max_Px = max(self.max_Px, Drone_position[0])
        self.min_Px = min(self.min_Px, Drone_position[0])
        self.angle_sum += Drone_position[0]

        if done:
            self.stop_drone()

        self.max_s = max(self.max_s, np.max(input_state))
        self.min_s = min(self.min_s, np.min(input_state))

        #fin_state = np.concatenate((np.array(input_state), [action_right, action_left]), axis=0)

        #return fin_state, reward, done, Drone_position
        return np.array(input_state), reward, done, Drone_position


    def step(self, actions):
        a_front = actions[0]   # a_flap : -1 ~ 1
        a_tail = actions[1]

        action_tail = (a_tail) * 120 # action : real number between -120~120, motor power
        if abs(action_tail) < 35:
           action_tail = 0

        action_front = ((a_front + 1) / 2.0) * 200 + 50 # action : real number between 0 ~ 250, motor power
        if action_front < 55:
           action_front = 0

        action_str = "L" + str(int(action_tail)) + "%" + "R" + str(int(action_front)) + "%"

        self.step_count += 1

        self.serial_channel.serialConnection.write(action_str.encode())
        for i in range(int(1280 * self.config["decision_period"])):
            self.action_right_queue.append(a_front)
            self.action_left_queue.append(a_tail)


    def calc_reward_done(self, prev_drone_position):
        done = False
        Drone_position = self.streamingClient.pos
        Drone_rotation = self.streamingClient.rot

        mu = [0, 300, 100]
        cov = [[10000, 0, 0], [0, 5000, 0], [0, 0, 10000]]
        rv = multivariate_normal(mu, cov)
        #
        # mu_rot = [0, 0]
        # cov_rot = [3000, 3000]
        # rv_rot = multivariate_normal(mu_rot, cov_rot)
        #
        reward = rv.pdf([Drone_position[0]*1000, Drone_position[1]*1000, Drone_position[2]*1000])*10**6
        if Drone_position[1]*1000 < 240:
            reward = 0
        # reward = (-(abs(Drone_position[0])+abs(Drone_position[2]))+(Drone_position[1]-0.19)*2)/100

        if self.step_count >= self.config["max_episode_len"]:
            done = True

        return reward, done, Drone_position


    def stop_drone(self):
        action_str = "L" + str(0) + "%" + "R" + str(0) + "%"
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


