import numpy as np
from utils.serialChannel import serialPlot
import copy
import math
from scipy.stats import multivariate_normal

from collections import deque


class Environment:
    def __init__(self, config, dw_thread, serial_channel: serialPlot):
        self.config = config
        self.is_discrete = config["is_discrete"]
        self.target_position = config["target_position"]
        self.dw_thread = dw_thread
        self.serial_channel = serial_channel
        self.step_count = 0
        self.warning_count = 0
        self.angle_sum = 0
        self.cur_angle = 0
        self.distance = 0
        self.reward = 0
        if config["is_discrete"]:
            self.action_count = np.zeros(config["n_action"])


    def reset(self):
        self.stop_drone()

        self.step_count = 0
        self.warning_count = 0
        if self.is_discrete:
            self.action_count = np.zeros(self.config["n_action"])
        self.distance_sum = 0
        self.cur_angle = 0
        self.max_angle, self.min_angle = -100000.0, 1000000.0
        self.max_s, self.min_s = -100000.0, 1000000.0


    def get_current_state(self):
        input_state = self.dw_thread.state

        reward, done, distance, Drone_position = self.calc_reward_done()
        self.max_angle = max(self.max_angle, distance)
        self.min_angle = min(self.min_angle, distance)
        self.distance_sum += distance
        if done:
            self.stop_drone()

        # print(self.step_count, np.array([input_state]))
        self.max_s = max(self.max_s, np.max(input_state))
        self.min_s = min(self.min_s, np.min(input_state))

        return np.array(input_state), reward, done, Drone_position


    def step(self, actions):
        [a_flap, a_tail] = actions   # a_flap : -1 ~ 1  ,  a_tail : -1 ~  1
        self.step_count += 1

        if self.is_discrete:
            if a_flap == 0:  # no action
                self.serial_channel.serialConnection.write("S0%".encode())

            elif a_flap == 1:  # medium force
                self.serial_channel.serialConnection.write("S150%".encode())

            elif a_flap == 2:  # medium force
                self.serial_channel.serialConnection.write("S200%".encode())

            elif a_flap == 3:  # medium force
                self.serial_channel.serialConnection.write("S250%".encode())

            if a_tail == 0:
                self.serial_channel.serialConnection.write("T0%".encode())

            if a_tail == 1:
                self.serial_channel.serialConnection.write("T200%".encode())

            if a_tail == 2:
                self.serial_channel.serialConnection.write("T-200%".encode())

            self.action_count += 1

        else:
            action_flap = ((a_flap + 1) / 2.0) * 250  # action : real number between -1 ~ 1, left wing
            action_tail = ((a_tail + 1) / 2.0) * 250  # right wing
            action_str = "S" + str(int(action_flap)) + "%" + "T" + str(int(action_tail)) + "%"
            self.serial_channel.serialConnection.write(action_str.encode())


    def forward_kinematics(self, angle_1, angle_2, angle_3):
        Px = -250
        Py = 380
        Pz = 220
        L1 = 60
        L2 = 380
        L3 = 250
        L4 = 160
        theta_1 = math.radians(angle_1)
        theta_2 = math.radians(angle_2)
        theta_3 = math.radians(angle_3)

        Px_drone = L3*(math.sin(theta_1)*math.cos(theta_2)*math.sin(theta_3)+math.cos(theta_1)*math.cos(theta_3))+L4*math.sin(theta_1)*math.sin(theta_2)-L2*math.sin(theta_1)*math.cos(theta_2)+Px
        Py_drone = L3*(math.cos(theta_1)*math.cos(theta_2)*math.sin(theta_3)-math.sin(theta_1)*math.cos(theta_3))+L4*math.cos(theta_1)*math.sin(theta_2)-L2*math.cos(theta_1)*math.cos(theta_2)+Py
        Pz_drone = L3*(math.sin(theta_2)*math.sin(theta_3))-L4*math.cos(theta_2)-L2*math.sin(theta_2)+Pz-L1

        Drone_position = [Px_drone, Py_drone, Pz_drone]

        return Drone_position


    def calc_reward_done(self):
        done = False
        # angle = copy.deepcopy(self.dw_thread.angle)
        # angle_1 = angle[0][-1]
        # angle_2 = angle[1][-1]
        # angle_3 = angle[2][-1]
        # Drone_position = self.forward_kinematics(angle_1, angle_2, angle_3)

        Drone_position = copy.deepcopy(self.dw_thread.drone_position)
        Drone_position = [Drone_position[0][-1],Drone_position[1][-1],Drone_position[2][-1]]
        # print(Drone_position)

        distance = math.sqrt((Drone_position[0]-self.target_position[0])**2 + (Drone_position[1]-self.target_position[1])**2 + (Drone_position[2]-self.target_position[2])**2)
        # print(self.distance)
        # mu = self.target_position
        # cov = [[100000, 0, 0], [0, 50000, 0], [0, 0, 100000]]
        # rv = multivariate_normal(mu, cov)
        # reward = rv.pdf(Drone_position)*10**7

        reward = (500 - abs(self.target_position[0] - Drone_position[0]))/3000 + (100 - abs(self.target_position[1] - Drone_position[1]))/2000
        # reward = (100 - abs(self.target_position[1] - Drone_position[1])) / 1000
        # print(self.reward)

        if self.step_count >= self.config["max_episode_len"]:
            done = True

        return reward, done, distance, Drone_position


    def stop_drone(self):
        self.serial_channel.serialConnection.write("S0%".encode())
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


