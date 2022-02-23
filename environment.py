import numpy as np
from utils.serialChannel import serialPlot
import copy
import math
from scipy.stats import multivariate_normal

from collections import deque


class Environment:
    def __init__(self, config, dw_thread, serial_channel: serialPlot):
        self.config = config
        self.target_position = config["target_position"]
        self.dw_thread = dw_thread
        self.serial_channel = serial_channel
        self.step_count = 0
        self.angle_sum = 0
        self.cur_angle = 0
        self.Drone_position = 0
        self.init_angle = 0


    def reset(self):
        self.stop_drone()
        self.step_count = 0
        self.cur_angle = 0
        self.max_angle, self.min_angle = -100000.0, 1000000.0
        self.max_s, self.min_s = -100000.0, 1000000.0


    def get_current_state(self):
        input_state = self.dw_thread.state

        reward, done, Drone_position = self.calc_reward_done()
        self.max_angle = max(self.max_angle, Drone_position)
        self.min_angle = min(self.min_angle, Drone_position)
        self.angle_sum += Drone_position

        if done:
            self.stop_drone()
            self.angle_sum = 0

        self.max_s = max(self.max_s, np.max(input_state))
        self.min_s = min(self.min_s, np.min(input_state))

        return np.array(input_state), reward, done, Drone_position


    def step(self, actions):
        a_flap = actions   # a_flap : -1 ~ 1
        self.step_count += 1
        action_flap = ((a_flap + 1) / 2.0) * 200 + 50  # action : real number between 0 ~ 250, motor power
        if action_flap < 55:
            action_flap = 0
        action_str = "T" + str(int(action_flap)) + "%"
        self.serial_channel.serialConnection.write(action_str.encode())


    def calc_reward_done(self):
        done = False
        Drone_angle = copy.deepcopy(self.dw_thread.drone_angle)
        Drone_position = self.init_angle + Drone_angle[0]
        # print(Drone_position)

        # if Drone_position > self.target_position + 90:
        #     reward = - abs(self.target_position + 90 - Drone_position) / 2000
        # else:
        #     reward = (90 - abs(self.target_position - Drone_position)) / 2000
        reward = (self.target_position - abs(self.target_position - Drone_position)) / 1000
        # print(reward)

        if self.step_count >= self.config["max_episode_len"]:
            done = True

        return reward, done, Drone_position


    def stop_drone(self):
        self.serial_channel.serialConnection.write("T0%".encode())


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


