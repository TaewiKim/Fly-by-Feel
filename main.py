from utils.dwclient import get_dewe_thread
from utils.serialChannel import serialPlot
from environment import Environment, DummyEnv
from model import Qnet, QnetConv
import torch
from replayBuffer import ReplayBuffer
import time, os
import pylab
from tensorboardX import SummaryWriter
from datetime import datetime, timedelta
from utils.util import save_config, save_model, write_summary



def main(config):
    save_config(config)

    s_channel = serialPlot('COM3', 19200, 4)  # dataNumBytes 4  : number of bytes of 1 data point
    s_channel.readSerialStart()  # starts background thread

    my_thread, tn, ready = get_dewe_thread()
    my_thread.start()
    tn.write(b"STARTTRANSFER 8001\r\n")
    print(tn.read_some())
    ready.wait()
    tn.write(b"STARTACQ\r\n")

    writer = SummaryWriter(logdir=config["log_dir"])

    env = Environment(config, my_thread, s_channel)
    # env = DummyEnv()
    q = QnetConv(config["learning_rate"], config["gamma"])
    q_target = QnetConv(config["learning_rate"], config["gamma"])
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer(config["buffer_limit"])

    time.sleep(2) # for waiting dw thread to be ready

    score = 0.0
    avg_loss = 0.0

    for n_epi in range(5000):
        epsilon = max(config["fin_eps"], config["init_eps"] - 0.01 * (n_epi))  # Linear annealing from 8% to 1%
        env.reset()
        done = False
        step = 0
        loop_t = 0.0
        prev_s, prev_a = None, None

        while not done:
            t1 = time.time()
            s, r, done = env.get_current_state()
            a = q.sample_action(torch.from_numpy(s).float().unsqueeze(1), epsilon)
            # a = 0
            env.step(a)

            done_mask = 0.0 if done else 1.0
            if prev_s is not None:
                memory.put((prev_s, prev_a, r, s, done_mask))
            prev_s, prev_a = s, a

            step += 1
            score += r
            if done:
                break

            t2 = time.time()-t1
            loop_t += t2

            if t2 < config["decision_period"]:
                time.sleep(config["decision_period"]-t2)

            # if step % 10 == 0:
            #     print(step, score, a)

        train_t = 0.0
        if memory.size() > config["train_start_buffer_size"]:
            avg_loss, train_t = q.train_net(q_target, memory, config["batch_size"])

        if n_epi != 0:
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                n_epi, score, memory.size(), epsilon * 100))
            write_summary(writer, n_epi, score, q.optimization_step, avg_loss, epsilon, env, loop_t/float(step), train_t)

        if n_epi % config["target_update_interval"] == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())

        if n_epi % config["model_save_interval"] == 0:
            save_model(config, q)

        score = 0.0


if __name__ == "__main__":
    config = {
        "buffer_limit" : 3000,
        "gamma" : 0.98,
        "learning_rate" : 0.0001,
        "print_interval" : 1,
        "target_update_interval": 3,
        "batch_size" : 32,
        "init_eps" : 0.4,
        "fin_eps" : 0.03,
        "train_start_buffer_size" : 1000,
        "decision_period" : 0.05,
        "model_save_interval" : 10,
        "max_episode_len" : 200, # 0.05*200 = 10 sec
        "log_dir" : "logs/" + datetime.now().strftime("[%m-%d]%H.%M.%S"),
    }
    main(config)