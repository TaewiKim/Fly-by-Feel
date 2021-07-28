from utils.dwclient import get_dewe_thread
from utils.serialChannel import serialPlot
from environment import Environment, DummyEnv
from models.dqn_model import Qnet, QnetConv
import torch
from replayBuffer import ReplayBuffer
import time, os
import pylab
from tensorboardX import SummaryWriter
from datetime import datetime, timedelta
from utils.util import save_config, save_dqn_model as save_model, write_summary
import numpy as np




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

    n_epi = 0
    writer = SummaryWriter(logdir=config["log_dir"])
    env = Environment(config, my_thread, s_channel)
    # env = DummyEnv()
    q = QnetConv(config["learning_rate"], config["gamma"], config["n_action"])

    if config["trained_model_path"]:
        checkpoint = torch.load(config["trained_model_path"])
        q.optimization_step = checkpoint['optimization_step']
        q.load_state_dict(checkpoint['model_state_dict'])
        q.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'n_epi' in checkpoint:
            n_epi = checkpoint['n_epi']
        print("Trained model", config["trained_model_path"], "suffessfully loaded")

    q_target = QnetConv(config["learning_rate"], config["gamma"], config["n_action"])
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer(config["buffer_limit"])

    time.sleep(2) # for waiting dw thread to be ready

    score = 0.0
    avg_loss = 0.0

    for i in range(2000):
        epsilon = max(config["fin_eps"], config["init_eps"] - 0.005 * (n_epi))  # Linear annealing from 8% to 1%
        env.reset()
        done = False
        step = 0
        loop_t = 0.0
        prev_s, prev_a = None, None
        init_t = time.time()

        while not done:
            t1 = time.time()
            s, r, done = env.get_current_state()
            a = q.sample_action(torch.from_numpy(s).float().unsqueeze(1), epsilon)   # for train
            # a = 0   # for no action
            # a = 3   # for max power
            # a = q.sample_action(torch.from_numpy(s).float().unsqueeze(1), 1.0)  # for random

            env.step(a)

            done_mask = 0.0 if done else 1.0
            if prev_s is not None:
                memory.put((prev_s, prev_a, r, s, done_mask))
            prev_s, prev_a = s, a

            step += 1
            score += r
            if done:
                print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                    n_epi, score, memory.size(), epsilon * 100))
                break

            t2 = time.time()-t1
            loop_t += t2

            if config["print_mode"]:
                print("epi:{}, step:{}, time:{:.3f}, angle:{:.2f}, action:{}".format(
                    i, step, time.time()-init_t, env.cur_angle, a))

            if t2 < config["decision_period"]:
                time.sleep(config["decision_period"]-t2)

        train_t = 0.0
        if memory.size() > config["train_start_buffer_size"]:
            avg_loss, train_t = q.train_net(q_target, memory, config["batch_size"])
            write_summary(writer, config, n_epi, score, q.optimization_step, avg_loss, epsilon, env, loop_t/float(step), train_t, 0.0)

        if n_epi % config["target_update_interval"] == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())

        if n_epi % config["model_save_interval"] == 0:
            save_model(config, q, n_epi)

        if n_epi % 30 == 0 and n_epi != 0:
            env.stop_drone()
            time.sleep(60)

        env.stop_drone()
        time.sleep(1)
        score = 0.0
        n_epi += 1

    env.stop_drone()

if __name__ == "__main__":
    config = {
        "is_discrete": True,
        "buffer_limit" : 3000,
        "gamma" : 0.98,
        "learning_rate" : 0.0001,
        "print_interval" : 1,
        "target_update_interval": 3,
        "batch_size" : 32,
        "init_eps" : 0.3,
        "fin_eps" : 0.03,
        "train_start_buffer_size" : 1000,
        "decision_period" : 0.05,
        "model_save_interval" : 20,
        "max_episode_len" : 200, # 0.05*200 = 10 sec
        "n_action" : 4,
        "log_dir" : "logs/" + datetime.now().strftime("[%m-%d]%H.%M.%S"),
        "trained_model_path" : None,
        "print_mode" : True,
        # "trained_model_path" : "logs/[07-19]baseline2/model_34720.tar",
        # "trained_model_path": "logs/[07-16]baseline1/model_19920.tar"
    }
    main(config)