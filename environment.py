import numpy as np
from utils.serialChannel import serialPlot
import copy

class Environment:
    def __init__(self, dw_thread, serial_channel: serialPlot):
        self.dw_thread = dw_thread
        self.serial_channel = serial_channel
        self.step_count = 0
        self.warning_count = 0

    def reset(self):
        self.step_count = 0
        self.warning_count = 0
        self.stop_drone()
        return np.array([self.dw_thread.channel_data]) / 17512291.0

    def step(self, action):
        self.step_count += 1

        if action == 0:  # no action
            self.stop_drone()

        elif action == 1:  # medium force
            self.serial_channel.serialConnection.write("S150%".encode())

        elif action == 2:  # strong force
            self.serial_channel.serialConnection.write("S300%".encode())

        next_state = np.array([copy.deepcopy(self.dw_thread.channel_data)])
        reward, done = self.calc_reward_done()

        if done:
            self.stop_drone()

        return next_state/ 17512291.0, reward, done

    def calc_reward_done(self):
        reward = 0
        done = False
        angle = self.serial_channel.getSerialData()

        if angle > 0:
            reward = angle/(90.0*100)
        if angle < -10:
            self.warning_count += 1

        if self.warning_count >= 500:
            done = True
            reward = -10

        if self.step_count >= 3000:
            done = True

        return reward, done

    def stop_drone(self):
        self.serial_channel.serialConnection.write("S0%".encode())


class DummyEnv:
    def __init__(self):
        self.step_count = 0

    def reset(self):
        return np.random.rand(32)

    def step(self, action):
        self.step_count += 1


        return np.random.rand(32), +0.01, False


