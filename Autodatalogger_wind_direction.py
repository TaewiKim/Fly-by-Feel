from utils.dwclient import get_dewe_thread
from utils.serialChannel import serialPlot
from environment import Environment, DummyEnv
from models.sac_model import PolicyNet, QNet, calc_target
import torch
from replayBuffer import ReplayBuffer
import time, os
import sys
from tensorboardX import SummaryWriter
from datetime import datetime, timedelta
from utils.util import save_config, save_sac_model as save_model, write_summary
import numpy as np
import pandas as pd
import random
from utils import dwserver

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.inf)

def main(config):
    np.set_printoptions(precision=3)
    save_config(config)

    s_channel = serialPlot('COM3', 19200, 4)  # dataNumBytes 4  : number of bytes of 1 data point
    s_channel.readSerialStart()  # starts background thread

    my_thread, tn, ready = get_dewe_thread()
    my_thread.start()
    tn.write(b"STARTTRANSFER 8001\r\n")
    print(tn.read_some())
    ready.wait()
    tn.write(b"STARTACQ\r\n")

    path = config["log_dir"]
    writer = SummaryWriter(logdir=path)
    env = Environment(config, my_thread, s_channel)
    # env = DummyEnv()

    time.sleep(5) # for waiting dw thread to be ready

    n_epi = 0
    Yaw = 0
    Pitch = 0
    Roll = 0
    # cases = np.array([[0, -90, -90], [0, -90, -45], [0, -90, 0], [0, 90, 45], [0, -90, 90], [0, -45, -90], [0, -45, -45], [0, -45, 0], [0, -45, 45], [0, -45, 90], [0, 0, -90], [0, 45, -45], [0, 45, 0], [0, 45, 45], [0, 90, -45], [0, 90, 0], [0, 90, 45], [180, -45, -90], [180, -45, -45], [180, -45, 0], [180, -45, 45], [180, -45, 90], [180, 0, -90], [180, 45, -45], [180, 45, 0], [180, 45, 45]])
    cases = np.array([
            [180, -60, -90],
            [180, -60, -60],
            [180, -60, -30],
            [180, -60, 0],
            [180, -60, 30],
            [180, -60, 60],
            [180, -60, 90],
            [180, -30, -90],
            [180, -30, -60],
            [180, -30, -30],
            [180, -30, 0],
            [180, -30, 30],
            [180, -30, 60],
            [180, -30, 90],
            [180, 0, -90],
            [180, 30, -60],
            [180, 30, -30],
            [180, 30, 0],
            [180, 30, 30],
            [180, 30, 60],
            [180, 60, -60],
            [180, 60, -30],
            [180, 60, 0],
            [180, 60, 30],
            [180, 60, 60],
            [180, 90, -60],
            [180, 90, -30],
            [180, 90, 0],
            [180, 90, 30],
            [180, 90, 60],
            [0, 0, -90],
            [0, 30, -60],
            [0, 30, -30],
            [0, 30, 0],
            [0, 30, 30],
            [0, 30, 60],
            [0, 60, -60],
            [0, 60, -30],
            [0, 60, 0],
            [0, 60, 30],
            [0, 60, 60],
            [0, 90, -90],
            [0, 90, -60],
            [0, 90, -30],
            [0, 90, 0],
            [0, 90, 30],
            [0, 90, 60],
            [0, 90, 90],
            [0, -60, -90],
            [0, -60, -60],
            [0, -60, -30],
            [0, -60, 0],
            [0, -60, 30],
            [0, -60, 60],
            [0, -60, 90],
            [0, -30, -90],
            [0, -30, -60],
            [0, -30, -30],
            [0, -30, 0],
            [0, -30, 30],
            [0, -30, 60],
            [0, -30, 90],
        ])

    drone_powers = [0, 0.5, 1]
    # drone_powers = [1]
    Fan_powers = [250, 200, 150]
    # Fan_powers = [250]

    for Yaw, Pitch, Roll in cases:
        Yaw_str = "Y" + str(Yaw) + "%"
        env.serial_channel.serialConnection.write(Yaw_str.encode())
        Pitch_str = "P" + str(Pitch) + "%"
        env.serial_channel.serialConnection.write(Pitch_str.encode())
        Roll_str = "R" + str(Roll) + "%"
        env.serial_channel.serialConnection.write(Roll_str.encode())

        data_log = [['Yaw', 'Pitch', 'Roll', 'Fan_Power', 'Drone_power', 'Right_wing', 'Left_wing']]

        for fan_power in Fan_powers:
            Fan_str = "F" + str(fan_power) + "%"
            env.serial_channel.serialConnection.write(Fan_str.encode())
            time.sleep(3)

            for drone_power in drone_powers:

                env.reset()
                done = False
                step = 0

                a_np = [drone_power]  # a_np = [-1., -1.]
                env.step(a_np)

                for i in range(5):
                    env.step(a_np)
                    s, r, done, drone_position = env.get_current_state(0)
                    time.sleep(0.1)

                while not done:
                    t1 = time.time()
                    s, r, done, drone_position = env.get_current_state(0)
                    env.step(a_np)

                    step += 1
                    if step >= config["max_episode_len"]:
                        done = True

                    if done:
                        print("Yaw :{}, Pitch : {}, Roll : {}, Drone_poewr : {}, Fan_power : {}".format(Yaw, Pitch, Roll, ((drone_power + 1) / 2.0) * 200 + 50, fan_power))
                        break

                    if config["print_mode"]:
                        data_log.append([Yaw, Pitch, Roll, fan_power])
                        data_log.extend(s)

                    t2 = time.time() - t1

                    if t2 < config["decision_period"]:
                        time.sleep(config["decision_period"]-t2)

                env.stop_drone()
                time.sleep(1)

        if config["print_mode"]:
            df = pd.DataFrame(data_log)
            df.to_csv(path + '/' + datetime.now().strftime("[%m-%d]%H.%M.%S") + "_Yaw_{}".format(
                Yaw) + "_Pitch_{}".format(Pitch) + "_Roll_{}".format(Roll) + '.csv')

        env.stop_drone()
        time.sleep(10)

    # for i in range(18):
    #     Yaw = i * 20 # Yaw: 0~360
    #     Yaw_str = "Y" + str(Yaw) + "%"
    #     env.serial_channel.serialConnection.write(Yaw_str.encode())
    #     time.sleep(2)
    #
    #     for j in range(9):
    #         Pitch = j * 20 - 90 # Pitch: -90~90
    #         Pitch_str = "P" + str(Pitch) + "%"
    #         env.serial_channel.serialConnection.write(Pitch_str.encode())
    #
    #         for k in range(9):
    #             Roll = k * 20 - 90 # Roll: -90~90
    #             Roll_str = "R" + str(Roll) + "%"
    #             env.serial_channel.serialConnection.write(Roll_str.encode())
    #             data_log = [['Yaw', 'Pitch', 'Roll', 'Fan_Power', 'Drone_power', 'Right_wing', 'Left_wing']]
    #
    #             for fan_power in Fan_powers:
    #                 Fan_str = "F" + str(fan_power) + "%"
    #                 env.serial_channel.serialConnection.write(Fan_str.encode())
    #                 time.sleep(3)
    #
    #                 for drone_power in drone_powers:
    #                     env.reset()
    #                     done = False
    #                     step = 0
    #
    #                     a_np = [drone_power]  # a_np = [-1., -1.]
    #                     env.step(a_np)
    #
    #                     time.sleep(1)
    #
    #                     while not done:
    #                         t1 = time.time()
    #                         s, r, done, drone_position = env.get_current_state(0)
    #                         env.step(a_np)
    #
    #                         step += 1
    #                         if step >= config["max_episode_len"]:
    #                             done = True
    #
    #                         if done:
    #                             print("Yaw :{}, Pitch : {}, Roll : {}, Drone_poewr : {}, Fan_power : {}".format(Yaw, Pitch, Roll, ((drone_power + 1) / 2.0) * 200 + 50, fan_power))
    #                             break
    #
    #                         if config["print_mode"]:
    #                             data_log.append([Yaw, Pitch, Roll, fan_power])
    #                             data_log.extend(s)
    #
    #                         t2 = time.time() - t1
    #
    #                         if t2 < config["decision_period"]:
    #                             time.sleep(config["decision_period"]-t2)
    #
    #                     env.stop_drone()
    #             if config["print_mode"]:
    #                 df = pd.DataFrame(data_log)
    #                 df.to_csv(path + '/' + datetime.now().strftime("[%m-%d]%H.%M.%S") + "_Yaw_{}".format(
    #                     Yaw) + "_Pitch_{}".format(Pitch) + "_Roll_{}".format(Roll) + '.csv')
    #
    #         env.stop_drone()
    #         # time.sleep(60)

    env.stop_drone()
    env.serial_channel.serialConnection.write("F0%".encode())

if __name__ == "__main__":
    config = {
        "target_position" : 0,
        "print_interval" : 1,
        "decision_period" : 0.05,
        "max_episode_len" : 205, # 0.05*200 = 10 sec
        "log_dir" : "wind_logs/" + datetime.now().strftime("[%m-%d]%H.%M.%S"),
        "print_mode": True,
    }
    main(config)