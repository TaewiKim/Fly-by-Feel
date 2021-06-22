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

    # for i in range(100):
    #     state_tensor = torch.from_numpy(state).float().unsqueeze(0)
    #     print(state_tensor.size())
    #     action = agent.sample_action(state_tensor, 0.5)
    #     print(action)
    #     next_state, reward, done = env.step(action)
    #     time.sleep(1)
    #     next_state, reward, done = env.step(0)
    #     time.sleep(1)
    #     state = next_state

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




    # num_episode = 2000
    #
    # for e in range(num_episode):
    #     time.sleep(2)
    #     s.warning = 0
    #     done = False
    #     score = 0
    #     score_avg = 0
    #     count = 0
    #     monitoring = []
    #     # thread = Thread(target=s.getSerialData)
    #     # thread.start()
    #     state = np.array([my_thread.channel_data, my_thread.channel_data])
    #     state = np.reshape(state, [1, -1, state_size, 1])
    #
    #     while not done:
    #         # 현재 상태로 행동을 선택
    #         action = agent.get_action(state)
    #         # 선택한 행동으로 환경에서 한 타임스텝 진행
    #         next_state_part, reward, done = agent.step(my_thread, s, action)
    #         next_state = np.array([next_state_part, next_state_part])
    #         # monitoring.append(next_state)
    #         next_state = np.reshape(next_state, [1, -1, state_size, 1])
    #
    #         # 타임스텝마다 보상 0.1, 에피소드가 중간에 끝나면 -100 보상
    #         score += reward
    #
    #         # 리플레이 메모리에 샘플 <s,a,r,s'> 저장
    #         agent.append_sample(state, action, reward, next_state, done)
    #         # print(state, ',', action, ',', reward, ',', next_state, ',', done)
    #         # 메모리에 데이터 1000개 이상 누적되면 학습 시작
    #
    #         if len(agent.memory) >= agent.train_start:
    #             agent.train_model()
    #
    #         # print(s.angle)
    #         state = next_state
    #         count += 1
    #
    #         if count >= 1000:
    #             PWM = 0
    #             MtrSpd = 'S' + str(PWM) + '%'  # '%' is our ending marker
    #             s.serialConnection.write(MtrSpd.encode())
    #             done = True
    #
    #         if done:
    #             # 각 에피소드마다 타깃 모델을 모델의 가중치로 업데이트
    #             agent.update_target_model()
    #             # 에피소드마다 학습 결과 출력
    #             score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
    #             print(
    #                 "episode: {:3d} | score: {:3.2f} | memory length: {:4d} | epsilon: {:.4f} | trial: {:3d}".format(e, score,
    #                                                                                                       len(
    #                                                                                                           agent.memory),
    #                                                                                                       agent.epsilon, count))
    #
    #             # 이동 평균이 400 이상 때 종료
    #             if score_avg > 50000:
    #                 agent.model.save_weights("./data", save_format="tf")
    #                 sys.exit()
    #         s.isRun = False
    # s.close()

if __name__ == "__main__":
    main()