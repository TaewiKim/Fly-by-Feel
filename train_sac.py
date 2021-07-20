from utils.dwclient import get_dewe_thread
from utils.serialChannel import serialPlot
from environment import Environment, DummyEnv
from models.sac_model import PolicyNet, QNet
import torch
from replayBuffer import ReplayBuffer
import time, os
import pylab
from tensorboardX import SummaryWriter
from datetime import datetime, timedelta
from utils.util import save_config, save_model, write_summary
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

    lr_q, tau = config["lr_q"], config["tau"]
    q1, q2, q1_target, q2_target = QNet(lr_q, tau), QNet(lr_q, tau), QNet(lr_q, tau), QNet(lr_q, tau)
    pi = PolicyNet(config["lr_pi"], config["init_alpha"], config["lr_alpha"])

    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())

    memory = ReplayBuffer(config["buffer_limit"])

    time.sleep(2) # for waiting dw thread to be ready

    score = 0.0
    avg_loss = 0.0

    for i in range(2000):
        env.reset()
        done = False
        step = 0
        loop_t = 0.0
        prev_s, prev_a = None, None

        while not done:
            t1 = time.time()
            s, r, done = env.get_current_state()
            # a, _ = pi(torch.from_numpy(s).float().unsqueeze(1))
            a = torch.tensor([0.])
            env.step(a)

            done_mask = 0.0 if done else 1.0
            if prev_s is not None:
                memory.put((prev_s, prev_a.item(), r, s, done_mask))
            prev_s, prev_a = s, a

            step += 1
            score += r
            if done:
                n_epi += 1
                break

            t2 = time.time()-t1
            loop_t += t2

            if t2 < config["decision_period"]:
                time.sleep(config["decision_period"]-t2)

            # if step % 10 == 0:
            #     print(step, score, a)

        train_t = 0.0
        if memory.size() > config["train_start_buffer_size"]:
            train_t_lst, loss_lst = [], []
            for i in range(20):
                t1 = time.time()
                mini_batch = memory.sample(batch_size)
                td_target = calc_target(pi, q1_target, q2_target, mini_batch, config["gamma"])
                loss1 = q1.train_net(td_target, mini_batch)
                loss2 = q2.train_net(td_target, mini_batch)
                entropy = pi.train_net(q1, q2, mini_batch)
                q1.soft_update(q1_target)
                q2.soft_update(q2_target)

                train_t_lst.append(time.time()-t1)
                loss_lst.append(loss1)

            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                n_epi, score, memory.size(), epsilon * 100))
            write_summary(writer, config, n_epi, score, q.optimization_step, np.mean(loss_lst), epsilon, env, loop_t/float(step), np.mean(train_t_lst))

        if n_epi % config["target_update_interval"] == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())

        if n_epi % config["model_save_interval"] == 0:
            save_model(config, q, n_epi)

        if n_epi % 30 == 0:
            env.stop_drone()
            time.sleep(60)


        env.stop_drone()
        time.sleep(1)
        score = 0.0

    env.stop_drone()

if __name__ == "__main__":
    config = {
        "is_discrete": False,
        "buffer_limit" : 3000,
        "gamma" : 0.98,
        "lr_pi" : 0.0005,
        "lr_q": 0.001,
        "init_alpha"  : 0.01,
        "print_interval" : 1,
        "target_update_interval": 3,
        "tau" : 0.01,
        "target_entropy" : -1.0,
        "lr_alpha" : 0.001,
        "batch_size" : 32,

        "train_start_buffer_size" : 1000,
        "decision_period" : 0.05,
        "model_save_interval" : 20,
        "max_episode_len" : 200, # 0.05*200 = 10 sec

        "log_dir" : "logs/" + datetime.now().strftime("[%m-%d]%H.%M.%S"),
        "trained_model_path" : "logs/[07-19]baseline2/model_34720.tar",
        # "trained_model_path": "logs/[07-16]baseline1/model_19920.tar"
    }
    main(config)