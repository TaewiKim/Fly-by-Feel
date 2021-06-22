import random
import numpy as np
from utils.dwclient import *
from utils.serialChannel import *
from environment import Environment
from model import Qnet
import torch
from replayBuffer import ReplayBuffer


def main():
    portName = 'COM3'
    baudRate = 19200
    dataNumBytes = 4  # number of bytes of 1 data point
    s_channel = serialPlot(portName, baudRate, dataNumBytes)
    s_channel.readSerialStart()  # starts background thread

    my_thread, tn, ready = get_dewe_thread()
    my_thread.start()
    tn.write(b"STARTTRANSFER 8001\r\n")
    print(tn.read_some())
    ready.wait()
    tn.write(b"STARTACQ\r\n")

    buffer_limit = 3000
    gamma = 0.98
    learning_rate = 0.0001
    print_interval = 20
    batch_size = 32

    env = Environment(my_thread, s_channel)
    q = Qnet(learning_rate, gamma)
    q_target = Qnet(learning_rate, gamma)
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer(buffer_limit)

    time.sleep(2) # for waiting dw thread to be ready
    s = env.reset()

    score = 0.0

    for n_epi in range(10000):
        epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))  # Linear annealing from 8% to 1%
        s = env.reset()
        done = False

        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            s_prime, r, done = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r / 100.0, s_prime, done_mask))
            s = s_prime

            score += r
            if done:
                break

        if memory.size() > 2000:
            q.train_net(q_target, memory, batch_size)

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                n_epi, score / print_interval, memory.size(), epsilon * 100))
            score = 0.0
    env.close()


if __name__ == "__main__":
    main()