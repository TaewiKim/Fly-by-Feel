from utils.dwclient import get_dewe_thread
from utils.serialChannel import serialPlot
from environment_230819 import Environment, DummyEnv
from models.sac_model_230819 import PolicyNet, QNet, calc_target
import torch
from replayBuffer import ReplayBuffer
import time, os
import sys
from tensorboardX import SummaryWriter
from datetime import datetime, timedelta
from utils.util_230819 import save_config, save_sac_model as save_model, write_summary
import numpy as np
import pandas as pd
import random
from utils.NatNetClient import NatNetClient
import pickle

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.inf)

def main(config):
    np.set_printoptions(precision=3)
    save_config(config)

    s_channel = serialPlot('COM27', 19200, 4)  # dataNumBytes 4  : number of bytes of 1 data point
    s_channel.readSerialStart()  # starts background thread

    my_thread, tn, ready = get_dewe_thread()
    my_thread.start()
    tn.write(b"STARTTRANSFER 8001\r\n")
    print(tn.read_some())
    ready.wait()
    tn.write(b"STARTACQ\r\n")

    def receiveNewFrame(frameNumber, markerSetCount, unlabeledMarkersCount, rigidBodyCount, skeletonCount,
                        labeledMarkerCount, timecode, timecodeSub, timestamp, isRecording, trackedModelsChanged):
        # print( "Received frame", frameNumber )
        None

    # This is a callback function that gets connected to the NatNet client. It is called once per rigid body per frame
    def receiveRigidBodyFrame(id, position, rotation):
        # print( "Received frame for rigid body", id )
        # print( "Received frame for rigid body", position, rotation)
        x, y, z = position
        r1, r2, r3, r4 = rotation
        # return x, y, z

    # This will create a new NatNet client
    streamingClient = NatNetClient()

    # Configure the streaming client to call our rigid body handler on the emulator to send data out.
    streamingClient.newFrameListener = receiveNewFrame
    streamingClient.rigidBodyListener = receiveRigidBodyFrame

    # Start up the streaming client now that the callbacks are set up.
    # This will run perpetually, and operate on a separate thread.
    streamingClient.run()

    n_epi = 0
    path = config["log_dir"]
    writer = SummaryWriter(logdir=path)
    env = Environment(config, my_thread, s_channel, streamingClient)
    # env = DummyEnv()

    lr_q, tau = config["lr_q"], config["tau"]
    q1, q2, q1_target, q2_target = QNet(lr_q, tau), QNet(lr_q, tau), QNet(lr_q, tau), QNet(lr_q, tau)
    pi = PolicyNet(config["lr_pi"], config["init_alpha"], config["lr_alpha"], config["target_entropy"])


    if config["trained_model_path"]:
        checkpoint = torch.load(config["trained_model_path"])
        pi.optimization_step = checkpoint['optimization_step']
        q1.load_state_dict(checkpoint['q1_state_dict'])
        q2.load_state_dict(checkpoint['q2_state_dict'])
        pi.load_state_dict(checkpoint['pi_state_dict'])

        q1.optimizer.load_state_dict(checkpoint['q1_optimizer_state_dict'])
        q2.optimizer.load_state_dict(checkpoint['q2_optimizer_state_dict'])
        pi.optimizer.load_state_dict(checkpoint['pi_optimizer_state_dict'])
        pi.log_alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        if 'n_epi' in checkpoint:
            n_epi = checkpoint['n_epi']
        print("Trained model", config["trained_model_path"], "suffessfully loaded")


    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())

    memory = ReplayBuffer(config["buffer_limit"])
    memory.buffer = pickle.load(open(config["buffer_load_path"], 'rb'))  # buffer 다시 불러올 때 주석 해제

    time.sleep(5) # for waiting dw thread to be ready

    Fan_str = "F" + str(config["Fan_power"]) + "%"
    env.serial_channel.serialConnection.write(Fan_str.encode())

    score = 0.0
    avg_loss = 0.0
    n_epi = 0
    action_sum = [0, 0]

    for i in range(config["max_episode_iter"]):
        env.reset()
        done = False
        step = 0
        data_log = [['epi', 'step', 'time', 'position', 'action', 'reward', 'state']]
        loop_t = 0.0
        prev_s, prev_a = None, None
        # prev_drone_position = [0, 0, 0]
        init_t = time.time()
        env.drone_shoot()
        env.Drone_position = np.array(streamingClient.pos)*1000

        while not done:
            t1 = time.time()
            s, r, done, drone_position = env.get_current_state()
            a, _ = pi(torch.from_numpy(s).float().unsqueeze(0))
            a_np = a.detach().numpy()
            a_np = a_np[0]
            print(a_np)

            if config["human_train"]:
                # a = env.human_action
                a = [0.8, a_np[0]]
                # a = [0.8, 0]
                a = np.array(a, dtype='float32')

            env.step(a)

            action_sum = [sum(x) for x in zip(action_sum, a_np)]

            done_mask = 0.0 if done else 1.0
            if prev_s is not None:
                memory.put((prev_s, a_np, r, s, done_mask))
            prev_s, prev_a = s, a_np
            # prev_drone_position = drone_position

            step += 1
            score += r

            if done:
                print("n_episode :{}, score : {:.1f}, n_buffer : {}".format(n_epi, score, memory.size()))
                break

            t2 = time.time()-t1
            loop_t += t2

            if config["print_mode"]:
                data_log.append([i, step, time.time()-init_t, drone_position, a_np, r, s[0], s[1]])


            if t2 < config["decision_period"]:
                time.sleep(config["decision_period"]-t2)

        train_t = 0.0
        env.stop_drone()

        pickle.dump(memory.buffer, open('./data/' + datetime.now().strftime("[%m-%d]%H.%M.%S") + '.pkl', 'wb'))

        if memory.size() > config["train_start_buffer_size"]:
            train_t_lst, loss_lst = [], []
            for i in range(20):
                t1 = time.time()
                mini_batch = memory.sample(config["batch_size"])
                td_target = calc_target(pi, q1_target, q2_target, mini_batch, config["gamma"])
                loss1 = q1.train_net(td_target, mini_batch)
                loss2 = q2.train_net(td_target, mini_batch)
                entropy = pi.train_net(q1, q2, mini_batch)
                q1.soft_update(q1_target)
                q2.soft_update(q2_target)

                train_t_lst.append(time.time()-t1)
                loss_lst.append(loss1)

            # write_summary(writer, config, n_epi, score, pi.optimization_step, np.mean(loss_lst), 0.0, env, loop_t/float(step), np.mean(train_t_lst), pi.log_alpha.exp().item(), action_sum, entropy.mean().item())
            write_summary(writer, config, n_epi, score, pi.optimization_step, np.mean(loss_lst), 0.0, env, loop_t/float(step), np.mean(train_t_lst), pi.log_alpha.exp().item(), action_sum, entropy.mean().item())
        if n_epi % config["model_save_interval"] == 0:
            save_model(config, q1, q2, pi, n_epi)

        if n_epi % 30 == 0 and n_epi != 0:
            env.stop_drone()

        if config["print_mode"]:
            df = pd.DataFrame(data_log)
            df.to_csv(path+'/'+ datetime.now().strftime("[%m-%d]%H.%M.%S")+"_epi_{}".format(n_epi)+'.csv')

        if config["Fan_rand"]:
            Fan_power = random.randint(200, 250)
            Fan_str = "F" + str(Fan_power) + "%"
            env.serial_channel.serialConnection.write(Fan_str.encode())

        env.stop_drone()
        env.shooter_back()
        user_input = input("Press any key to continue")

        score = 0.0
        n_epi += 1
        action_sum = [0, 0]
        env.Px_sum = 0
        env.Py_sum = 0
        env.Pz_sum = 0


    env.stop_drone()
    # env.serial_channel.serialConnection.write("F0%".encode())

if __name__ == "__main__":
    config = {
        "buffer_limit" : 10000,  #10000
        "buffer_load_path": "data/[08-28]21.38.14.pkl",
        "gamma" : 0.98,
        "lr_pi" : 0.0005, #0.0001
        "lr_q": 0.0005, #0.0001
        "init_alpha"  : 0.0005, #0.001
        "print_interval" : 1,
        "target_update_interval": 3,
        "tau" : 0.001,
        "target_entropy" : -1.0,
        "lr_alpha" : 0.0001, #0.001
        "batch_size" : 64, #32
        "train_start_buffer_size" : 500,  #5000
        "decision_period" : 0.05,
        "model_save_interval" : 10,
        "max_episode_len" : 40, # 0.05*40 = 2 sec
        "max_episode_iter": 500,
        "log_dir" : "logs/" + datetime.now().strftime("[%m-%d]%H.%M.%S"),
        "target_position": [-2000, 1500, 4000],
        "print_mode": True,
        "Fan_power": 0,
        "Fan_rand": False,
        "human_train": True,
        # "trained_model_path": None,
        #  "trained_model_path": "logs/[08-24]18.55.52/sac_model_0.tar"
        # # "trained_model_path": "logs/[08-18]19.52.20/sac_model_2000.tar"
        # "trained_model_path": "logs/[08-25]12.33.16_left/sac_model_320.tar" # left
        "trained_model_path": "logs/[08-28]20.20.21_right/sac_model_620.tar"
    }

    main(config)
