import numpy as np
from utils.serialChannel import serialPlot
import copy
from collections import deque


class Environment:
    def __init__(self, config, dw_thread, serial_channel: serialPlot):
        self.config = config
        self.dw_thread = dw_thread
        self.serial_channel = serial_channel
        self.step_count = 0
        self.warning_count = 0
        self.action_count = np.zeros(config["n_action"])
        self.angle_sum = 0


    def reset(self):
        self.stop_drone()

        self.step_count = 0
        self.warning_count = 0
        self.action_count = np.zeros(self.config["n_action"])
        self.angle_sum = 0
        self.max_s, self.min_s = -100000.0, 1000000.0


    def get_current_state(self):
        input_state = copy.deepcopy(self.dw_thread.state)

        reward, done, angle = self.calc_reward_done()
        self.angle_sum += angle
        if done:
            self.stop_drone()

        self.max_s = max(self.max_s, np.max(input_state))
        self.min_s = min(self.min_s, np.min(input_state))

        return np.array([input_state]), reward, done


    def step(self, action):
        self.step_count += 1

        if action == 0:  # no action
            self.stop_drone()

        elif action == 1:  # medium force
            self.serial_channel.serialConnection.write("S150%".encode())

        elif action == 2:  # medium force
            self.serial_channel.serialConnection.write("S200%".encode())

        elif action == 3:  # medium force
            self.serial_channel.serialConnection.write("S250%".encode())

        self.action_count[action] += 1

    def calc_reward_done(self):
        reward = 0
        done = False
        angle = self.serial_channel.getSerialData()

        if angle > 0:
            reward = angle/(20.0*100)
        if angle < -10:
            self.warning_count += 1

        if self.warning_count >= 500:
            done = True
            reward = -10

        if self.step_count >= self.config["max_episode_len"]:
            done = True

        return reward, done, angle

    def stop_drone(self):
        self.serial_channel.serialConnection.write("S0%".encode())

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


