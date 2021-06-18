import time
import numpy as np
import telnetlib
import sys
from threading import Event
import dwserver_mod2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import collections

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
        #self.current_min = float(input[21 + int(input[21]) + 1].replace(",", "."))
        #self.current_max = float(input[21 + int(input[21]) + 2].replace(",", "."))
        #self.current_avg = float(input[21 + int(input[21]) + 3].replace(",", "."))


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

# SETUP MUST BE OPENED IN DEWESOFT
ready = Event()
HOST = 'localhost' # if you want to connect remotly enter IP of pc
tn = telnetlib.Telnet(HOST, '8999') # 8999 is standard port number
print(tn.read_some())

tn.write(b"SETMODE 1\r\n") # we change to control mode
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

tn.write(b"LISTUSEDCHS\r\n") # here we get list of all used channels
list_of_used_ch = process_listusedchs(tn.read_until(b'+ETX end list\r\n'), float(sample_rate_str))
print(list_of_used_ch)

list_of_used_ch = prepare_channels([0], list_of_used_ch) # filter channels
tn.write(b'/stx preparetransfer\r\nCH 0\r\n/etx\r\n')
# tn.write(b'/stx preparetransfer\r\nCH 0\r\nCH 1\r\nCH 2\r\n/etx\r\n') # here we select which channels we want to transfer
print(tn.read_some())

my_thread = dwserver_mod2.MyThread(ready, list_of_used_ch)
my_thread.start()

tn.write(b"STARTTRANSFER 8001\r\n")
print(tn.read_some())
ready.wait()
tn.write(b"STARTACQ\r\n")
ready.set()

# plot = dwserver_mod2.DewePlot(ready, my_thread.channel_data)
# plt.show()

class DewePlot:
    def __init__(self, maxPlotLength):
        self.plotTimer = 0
        self.previousTimer = 0
        self.plotMaxLength = maxPlotLength
        self.data = collections.deque([0] * self.plotMaxLength, maxlen=self.plotMaxLength)

    def getSerialData(self, frame, lines, lineValueText, lineLabel, timeText):
        currentTimer = time.perf_counter()
        self.plotTimer = int((currentTimer - self.previousTimer) * 1000)     # the first reading will be erroneous
        self.previousTimer = currentTimer
        timeText.set_text('Plot Interval = ' + str(self.plotTimer) + 'ms')
        # value,  = struct.unpack('f', self.rawData)    # use 'h' for a 2 byte integer
        # self.data.append(value)    # we get the latest data point and append it to our array
        self.data.append(my_thread.channel_data)
        lines.set_data(range(self.plotMaxLength), my_thread.channel_data)
        lineValueText.set_text('[' + lineLabel + '] = ' + str(my_thread.channel_data))
        # self.csvData.append(self.data[-1])

    # def close(self):
    #     self.isRun = False
    #     self.thread.join()
    #     self.serialConnection.close()
    #     print('Disconnected...')
    #     # df = pd.DataFrame(self.csvData)
    #     # df.to_csv('/home/rikisenia/Desktop/data.csv')

# maxPlotLength = 1000
# d = DewePlot(maxPlotLength)
#
# pltInterval = 100  # Period at which the plot animation updates [ms]
# xmin = 0
# xmax = maxPlotLength
# ymin = -(170000)
# ymax = 170000
# fig = plt.figure()
# ax = plt.axes(xlim=(xmin, xmax), ylim=(float(ymin - (ymax - ymin) / 10), float(ymax + (ymax - ymin) / 10)))
# ax.set_title('DeweSoft Analog Read')
# ax.set_xlabel("time")
# ax.set_ylabel("AnalogRead Value")
#
# lineLabel = 'Resistance Value'
# timeText = ax.text(0.50, 0.95, '', transform=ax.transAxes)
# lines = ax.plot([], [], label=lineLabel)[0]
# lineValueText = ax.text(0.50, 0.90, '', transform=ax.transAxes)
# anim = animation.FuncAnimation(fig, d.getSerialData, fargs=(lines, lineValueText, lineLabel, timeText),
#                                interval=pltInterval)  # fargs has to be a tuple
#
# plt.legend(loc="upper left")
# plt.show()

while True:
    state = np.array([my_thread.channel_data,my_thread.channel_data])
    state = np.reshape(state, [1, -1, 2, 1])
    print(state)

    time.sleep(0.01)

my_thread.join()

