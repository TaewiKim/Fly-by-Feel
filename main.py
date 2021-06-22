import random
import numpy as np
from utils.dwclient import *
from utils.serialChannel import *
from environment import Environment
from model import Qnet
import torch
from replayBuffer import ReplayBuffer


def main(config):
    s_channel = serialPlot('COM3', 19200, 4)  # dataNumBytes 4  : number of bytes of 1 data point
    s_channel.readSerialStart()  # starts background thread

    my_thread, tn, ready = get_dewe_thread()
    my_thread.start()
    tn.write(b"STARTTRANSFER 8001\r\n")
    print(tn.read_some())
    ready.wait()
    tn.write(b"STARTACQ\r\n")

    env = Environment(my_thread, s_channel)
    q = Qnet(config["learning_rate"], config["gamma"])
    q_target = Qnet(config["learning_rate"], config["gamma"])
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer(config["buffer_limit"])

    time.sleep(2) # for waiting dw thread to be ready

    score = 0.0

    for n_epi in range(10000):
        epsilon = max(config["fin_eps"], config["init_eps"] - 0.01 * (n_epi / 20))  # Linear annealing from 8% to 1%
        s = env.reset()
        done = False

        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            # a = 0
            s_prime, r, done = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r / 100.0, s_prime, done_mask))
            s = s_prime

            score += r
            if done:
                break

        if memory.size() > config["train_start_buffer_size"]:
            q.train_net(q_target, memory, config["batch_size"])

        if n_epi % config["print_interval"] == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                n_epi, score / config["print_interval"], memory.size(), epsilon * 100))
            score = 0.0
    env.close()

if __name__ == "__main__":
    config = {
        "buffer_limit" : 3000,
        "gamma" : 0.98,
        "learning_rate" : 0.0001,
        "print_interval" : 20,
        "batch_size" : 32,
        "init_eps" : 0.5,
        "fin_eps" : 0.01,
        "train_start_buffer_size" : 2000
    }
    main(config)