import numpy as np
from utils.serialChannel import serialPlot
import copy
from collections import deque


class Environment:
    def __init__(self, config, dw_thread, serial_channel: serialPlot):
        self.config = config
        self.is_discrete = config["is_discrete"]
        self.dw_thread = dw_thread
        self.serial_channel = serial_channel
        self.step_count = 0
        self.warning_count = 0
        self.angle_sum = 0
        if config["is_discrete"]:
            self.action_count = np.zeros(config["n_action"])


    def reset(self):
        self.stop_drone()

        self.step_count = 0
        self.warning_count = 0
        if self.is_discrete:
            self.action_count = np.zeros(self.config["n_action"])
        self.angle_sum = 0
        self.max_angle, self.min_angle = -100000.0, 1000000.0
        self.max_s, self.min_s = -100000.0, 1000000.0


    def get_current_state(self):
        input_state = copy.deepcopy(self.dw_thread.state)

        reward, done, angle = self.calc_reward_done()
        self.max_angle = max(self.max_angle, angle)
        self.min_angle = min(self.min_angle, angle)
        self.angle_sum += angle
        if done:
            self.stop_drone()

        # print(self.step_count, np.array([input_state]))
        self.max_s = max(self.max_s, np.max(input_state))
        self.min_s = min(self.min_s, np.min(input_state))

        return np.array([input_state]), reward, done


    def step(self, action):
        self.step_count += 1

        if self.is_discrete:
            if action == 0:  # no action
                self.stop_drone()

            elif action == 1:  # medium force
                self.serial_channel.serialConnection.write("S150%".encode())

            elif action == 2:  # medium force
                self.serial_channel.serialConnection.write("S200%".encode())

            elif action == 3:  # medium force
                self.serial_channel.serialConnection.write("S250%".encode())

            self.action_count[action] += 1

        else:
            action_power = ((action+1)/2.0) * 250  # action : real number between -1 ~ 1
            print(action_power, action)
            action_str = "S" + str(int(action_power)) + "%"
            self.serial_channel.serialConnection.write(action_str.encode())

    def calc_reward_done(self):
        done = False
        angle = self.serial_channel.getSerialData()
        reward = angle/(10.0*100)

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


