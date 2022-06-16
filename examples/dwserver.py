'''
    Simple socket server using threads
'''

import socket
import sys
import threading
import struct
import copy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time
import collections
import math
from mpl_toolkits.mplot3d import Axes3D

HOST = ''  # Symbolic name meaning all available interfaces
PORT = 8001  # Dewesoft port for data stream


#How many samples of synchronous channels are presented
MAX_SYNC_SAMPLES = 50000
#How many samples of asynchronous channels are presented
MAX_ASYNC_SAMPLES = 50000

class Package:
    def __init__(self, input):
        self.start_index = input.find(b'\x00\x01\x02\x03\x04\x05\x06\x07')
        self.end_index = input.find(b'\x07\x06\x05\x04\x03\x02\x01\x00')
        if self.start_index == -1 or self.end_index == -1:
            self.full_package = False
            return

        self.packet_size = struct.unpack('i', input[self.start_index + 8:self.start_index + 12])[0]
        self.packet_type = struct.unpack('i', input[self.start_index + 12:self.start_index + 16])[0]
        self.samples_in_packet = struct.unpack('i', input[self.start_index + 16:self.start_index + 20])[0]
        self.samples_acquired_so_far = struct.unpack('q', input[self.start_index + 20:self.start_index + 28])[0]
        self.absolute_relative_time = struct.unpack('d', input[self.start_index + 28:self.start_index + 36])[0]
        self.data = copy.deepcopy(input[self.start_index + 36:self.end_index])
        self.full_package = True

    def read_data_as_async(self, data_type, data_type_size):
        num_of_samples = struct.unpack('i', self.data[0:4])[0]
        data = []
        timestamp = []
        for i in range(1, num_of_samples + 1):
            data.append(struct.unpack(data_type, self.data[i * 4:i * 4 + data_type_size])[0])
        for i in range(1, num_of_samples + 1):
            timestamp.append(struct.unpack('d', self.data[i * 4 + (i - 1) * 4 + num_of_samples * data_type_size: i * 4 + (i - 1) * 4 + 8 + num_of_samples * data_type_size])[0])
        self.data = self.data[4 + num_of_samples * (data_type_size + 8):]  # 8 is size of double timestamp
        #print(f'{data} {timestamp}')
        return data, timestamp

    def read_data_as_sync(self, data_type, data_type_size, list_of_used_ch):
        num_of_samples = struct.unpack('i', self.data[0:4])[0]
        # print(f'{num_of_samples} {data_type} {data_type_size}')
        result = [struct.unpack('f', self.data[i + 4: i + 4 + data_type_size])[0] for i in
                  range(0, num_of_samples * data_type_size, data_type_size)]
        self.data = self.data[4 + num_of_samples * data_type_size:]
        # print(result)
        return result

    def read_data_as_single_value(self, data_type, data_type_size):
        num_of_samples = struct.unpack('i', self.data[0:4])[0]
        result = [struct.unpack(data_type, self.data[4:4 + data_type_size])[0]]
        self.data = self.data[4 + data_type_size:]
        return result

    def fill(self, input):
        location = input.find(b'\x07\x06\x05\x04\x03\x02\x01\x00')
        if location != -1:
            self.full_package = True
            self.data = self.data + input[:location]
        else:
            self.full_package = False
            self.data = self.data + input[:]


class DewePlot:
    def __init__(self, ready, list_of_used_ch, Drone_position):
        self.ready = ready
        self.list_of_used_ch = list_of_used_ch
        self.Drone_position = Drone_position
        fig = plt.figure()
        ax = Axes3D(fig, auto_add_to_figure=False)
        line_ani = animation.FuncAnimation(fig, self.Drone_position, interval=50, blit=False)
        # line_ani.save(r'AnimationNew.mp4')
        plt.show()

        # if (len(self.list_of_used_ch[1].timestamp) == len(self.list_of_used_ch[1].channel_data)):
        #     #print(f'Async: {len(self.list_of_used_ch[1].timestamp)} {len(self.list_of_used_ch[1].channel_data)}')
        #     self.hLine2.set_data(self.list_of_used_ch[1].timestamp, self.list_of_used_ch[1].channel_data)
        #     self.hLine2.axes.relim()
        #     self.hLine2.axes.autoscale_view()

    def update_lines(num, dataLines, lines):
        for line, data in zip(lines, dataLines):
            # NOTE: there is no .set_data() for 3 dim data...
            line.set_data(data[0:2, :num])
            line.set_3d_properties(data[2, :num])
        return lines


class MyThread(threading.Thread):
    def __init__(self, ready, list_of_used_ch):
        threading.Thread.__init__(self)
        zeros = np.zeros(320)

        self.ready = ready
        self.list_of_used_ch = list_of_used_ch
        self.buffer_data = b''
        self.time = time.time()
        self.time_sync = 0
        self.state = [0, 0]
        self.angle = [0, 0, 0]
        self.Drone_position = [0, 0, 0]
        self.rightWing = collections.deque(maxlen=320)
        self.rightWing.extend(zeros)
        self.leftWing = collections.deque(maxlen=320)
        self.leftWing.extend(zeros)
        self.angle_1 = collections.deque(maxlen=320)
        self.angle_1.extend(zeros)
        self.angle_2 = collections.deque(maxlen=320)
        self.angle_2.extend(zeros)
        self.angle_3 = collections.deque(maxlen=320)
        self.angle_3.extend(zeros)

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

    def run(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print('Socket created')
        # Bind socket to local host and port
        try:
            s.bind((HOST, PORT))
        except socket.error as msg:
            print('Bind failed. Error Code : ' + str(msg.args[0]) + ' Message ' + msg.args[1])
            s.close()
            sys.exit()

        print('Socket bind complete')
        # Start listening on socket
        s.listen(10)
        print('Socket now listening')

        # now keep talking with the client
        # wait to accept a connection - blocking call
        conn, addr = s.accept()
        print('Connected with ' + addr[0] + ':' + str(addr[1]))
        self.ready.set()
        while True:
            self.buffer_data += conn.recv(4000)  # size 2000 is default size of dewesoft packages
            if not self.buffer_data:
                break

            current_package = Package(self.buffer_data)
            if not current_package.full_package:
                continue
            else:
                self.buffer_data = self.buffer_data[current_package.end_index + 8:]

                for i in range(0, len(self.list_of_used_ch)):
                    if self.list_of_used_ch[i].async_ch:
                        channel_data, timestamp = current_package.read_data_as_async(self.list_of_used_ch[i].data_type, self.list_of_used_ch[i].data_type_size)
                        print("hihi", channel_data)
                        if len(channel_data) > 0:
                            self.list_of_used_ch[i].channel_data = (self.list_of_used_ch[i].channel_data  + channel_data)[-MAX_ASYNC_SAMPLES:]
                            self.list_of_used_ch[i].timestamp = (self.list_of_used_ch[i].timestamp + timestamp)[-MAX_ASYNC_SAMPLES:]
                    elif self.list_of_used_ch[i].single_value:
                        self.list_of_used_ch[i].channel_data = current_package.read_data_as_single_value(self.list_of_used_ch[i].data_type, self.list_of_used_ch[i].data_type_size)
                    else:
                        channel_data = current_package.read_data_as_sync(self.list_of_used_ch[i].data_type, self.list_of_used_ch[i].data_type_size, self.list_of_used_ch[i])
                        self.list_of_used_ch[i].channel_data = (self.list_of_used_ch[i].channel_data + channel_data)[-MAX_SYNC_SAMPLES:]

                        time_increase =  1 / (self.list_of_used_ch[i].sample_rate / self.list_of_used_ch[i].sample_div)
                        first_time = self.list_of_used_ch[i].number_of_added_samples * time_increase
                        second_time = (self.list_of_used_ch[i].number_of_added_samples + len(channel_data)) *  time_increase
                        
                        self.list_of_used_ch[i].timestamp = (self.list_of_used_ch[i].timestamp + list(np.linspace(first_time,
                                                                                                                second_time, 
                                                                                                                num=len(channel_data))))[-MAX_SYNC_SAMPLES:]

                        self.list_of_used_ch[i].number_of_added_samples = self.list_of_used_ch[i].number_of_added_samples + len(channel_data)\

                        # print(len(channel_data), time.time()-self.time, channel_data)

                        self.time_sync = time.time() - self.time

                        if i == 0:
                            self.rightWing.extend(channel_data)
                            self.state[0] = self.rightWing

                        if i == 1:
                            self.leftWing.extend(channel_data)
                            self.state[1] = self.leftWing

                        if i == 2:
                            self.angle_1.extend(channel_data)
                            self.angle[0] = self.angle_1

                        if i == 3:
                            self.angle_2.extend(channel_data)
                            self.angle[1] = self.angle_2

                        if i == 4:
                            self.angle_3.extend(channel_data)
                            self.angle[2] = self.angle_3

                        self.Drone_position = self.forward_kinematics(self.angle_1[-1], self.angle_2[-1], self.angle_3[-1])
            print(self.Drone_position)

        s.close()
