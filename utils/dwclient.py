import telnetlib
import sys
from threading import Event
import utils.dwserver as dwserver
import matplotlib.pyplot as plt

DATA_TYPES = ['b', 'B', 'h', 'H', 'i', 'f', 'q', 'd']
DATA_SIZE = [1, 1, 2, 2, 4, 4, 8, 8]


class Channel:
    def __init__(self, input, sample_rate):
        self.ch = input[0]
        self.channel_type = input[1]
        self.num = int(input[2])
        self.name = input[3]
        self.desc = input[4]
        self.unit = input[5]
        self.timestamp = []
        self.channel_data = []
        self.number_of_added_samples = 0
        self.sample_rate = sample_rate

        self.async_ch = False
        self.single_value = False
        if input[6] == "Async":
            self.async_ch = True
        elif input[6] == "SingleValue":
            self.single_value = True
        else:
            self.sample_div = int(input[6])
        self.expected_async_rate = float(input[7])
        self.measur_type = int(input[8])
        self.data_type = DATA_TYPES[int(input[9])]
        self.data_type_size = DATA_SIZE[int(input[9])]
        self.buffer_size = int(input[10])
        self.custom_scale = float(input[11])
        self.custom_offset = float(input[12])
        self.scale_raw_data = float(input[13].replace(",", "."))
        self.offset_raw_data = float(input[14])
        self.description = input[15]
        self.settings = input[16]
        self.range_min = float(input[17].replace(",", "."))
        self.range_max = float(input[18].replace(",", "."))
        if input[19] == 'OvlYes':
            self.can_overload = True
        elif input[19] == 'OvlNo':
            self.can_overload = False
        else:
            self.can_overload = False
        self.auto_zero = bool(input[20])
        if int(input[21]) > 0:
            self.discrete_list = [element for element in input[22: 22 + int(21)]]
        else:
            self.discrete_list = []
        # self.current_min = float(input[21 + int(input[21]) + 1].replace(",", "."))
        # self.current_max = float(input[21 + int(input[21]) + 2].replace(",", "."))
        # self.current_avg = float(input[21 + int(input[21]) + 3].replace(",", "."))


def process_listusedchs(input, sample_rate):
    list_of_used_ch = []
    input = input.decode('utf-8').split('\r\n')[1:-2]
    for element in input:
        output_element = element.split('\t')
        list_of_used_ch.append(Channel(output_element, sample_rate))
    return list_of_used_ch


def prepare_channels(selected_channels, input_array):
    result_array = []
    for element in selected_channels:
        result_array.append(input_array[element])
    return result_array

def get_dewe_thread():

    # SETUP MUST BE OPENED IN DEWESOFT
    ready = Event()
    HOST = 'localhost'  # if you want to connect remotly enter IP of pc
    tn = telnetlib.Telnet(HOST, '8999')  # 8999 is standard port number
    tn.read_some()

    tn.write(b"SETMODE 1\r\n")  # we change to control mode
    print(tn.read_some())

    tn.write(b"GETSAMPLERATE\r\n")
    output = tn.read_some().decode('utf-8')

    start = output.find("+OK ") + len("+OK ")
    end = output.find("\r\n")
    sample_rate_str = output[start:end]

    tn.write(b"ISMEASURING\r\n")
    output = tn.read_some().decode('utf-8')
    if output == "+OK Yes\r\n":
        tn.write(b"STOP\r\n")
        tn.read_some()

    tn.write(b"LISTUSEDCHS\r\n")  # here we get list of all used channels
    list_of_used_ch = process_listusedchs(tn.read_until(b'+ETX end list\r\n'), float(sample_rate_str))

    list_of_used_ch = prepare_channels([1], list_of_used_ch)  # filter channels
    tn.write(
        b'/stx preparetransfer\r\nCH 0\r\nCH 1\r\nCH 2\r\n/etx\r\n')  # here we select which channels we want to transfer
    print(tn.read_some())

    my_thread = dwserver.MyThread(ready, list_of_used_ch)
    # my_thread.start()
    # tn.write(b"STARTTRANSFER 8001\r\n")
    # print(tn.read_some())
    # ready.wait()
    # tn.write(b"STARTACQ\r\n")

    # plot = dwserver.DewePlot(ready, list_of_used_ch)
    # plt.show()
    # my_thread.join()

    return my_thread, tn, ready

