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
        self.action_count = np.array([0,0,0])
        self.angle_sum = 0

        self.state = deque(maxlen=100)

    def reset(self):
        self.stop_drone()

        self.step_count = 0
        self.warning_count = 0
        self.action_count = np.array([0, 0, 0])
        self.angle_sum = 0
        self.state.extend(np.zeros(100))


    def get_current_state(self):
        input_state = copy.deepcopy(self.dw_thread.channel_data)
        self.state.extend(input_state)
        reward, done, angle = self.calc_reward_done()
        self.angle_sum += angle
        if done:
            self.stop_drone()

        return np.array([self.state]), reward, done


    def step(self, action):
        self.step_count += 1

        if action == 0:  # no action
            self.stop_drone()

        elif action == 1:  # medium force
            self.serial_channel.serialConnection.write("S50%".encode())

        elif action == 2:  # strong force
            self.serial_channel.serialConnection.write("S80%".encode())

        self.action_count[action] += 1

    def calc_reward_done(self):
        reward = 0
        done = False
        angle = self.serial_channel.getSerialData()


        if angle > 0:
            reward = angle/(50.0*100)
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


