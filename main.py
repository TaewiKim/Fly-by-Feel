from utils.dwclient import get_dewe_thread
from utils.serialChannel import serialPlot
from environment import Environment, DummyEnv
from model import Qnet
import torch
from replayBuffer import ReplayBuffer
import time, datetime



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
    # env = DummyEnv()
    q = Qnet(config["learning_rate"], config["gamma"])
    q_target = Qnet(config["learning_rate"], config["gamma"])
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer(config["buffer_limit"])

    time.sleep(2) # for waiting dw thread to be ready

    score = 0.0

    for n_epi in range(10000):
        epsilon = max(config["fin_eps"], config["init_eps"] - 0.01 * (n_epi))  # Linear annealing from 8% to 1%
        s = env.reset()
        done = False
        step = 0

        while not done:
            t1 = time.time()
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)

            if step % 10 == 0:
                print(step, score, a)
            # a = 0
            s_prime, r, done = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r, s_prime, done_mask))
            s = s_prime

            step += 1
            score += r
            if done:
                break

            t2 = time.time()-t1

            if t2<0.025:
                time.sleep(0.025-t2)

        if memory.size() > config["train_start_buffer_size"]:
            q.train_net(q_target, memory, config["batch_size"])

        if n_epi % config["print_interval"] == 0 and n_epi != 0:
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                n_epi, score / config["print_interval"], memory.size(), epsilon * 100))
            score = 0.0

        if n_epi % config["target_update_interval"] == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())

    env.close()

if __name__ == "__main__":
    config = {
        "buffer_limit" : 3000,
        "gamma" : 0.98,
        "learning_rate" : 0.0001,
        "print_interval" : 5,
        "target_update_interval": 2,
        "batch_size" : 64,
        "init_eps" : 1.0,
        "fin_eps" : 0.0,
        "train_start_buffer_size" : 2000
    }
    main(config)