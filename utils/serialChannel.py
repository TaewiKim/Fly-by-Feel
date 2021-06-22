import serial
from threading import Thread
import time
import collections
import struct
import copy

class serialPlot:
    def __init__(self, serialPort, serialBaud, dataNumBytes):
        numPlots=1
        self.port = serialPort
        self.baud = serialBaud
        self.dataNumBytes = dataNumBytes
        self.rawData = bytearray(numPlots * dataNumBytes)
        self.dataType = None
        if dataNumBytes == 2:
            self.dataType = 'h'  # 2 byte integer
        elif dataNumBytes == 4:
            self.dataType = 'f'  # 4 byte float
        self.angle = 0
        self.isRun = True
        self.isReceiving = False
        self.thread = None

        self.plotTimer = 0
        self.previousTimer = 0
        self.csvData = []
        self.buffer = collections.deque(maxlen=100)


        print('Trying to connect to: ' + str(serialPort) + ' at ' + str(serialBaud) + ' BAUD.')
        try:
            self.serialConnection = serial.Serial(serialPort, serialBaud, timeout=4)
            print('Connected to ' + str(serialPort) + ' at ' + str(serialBaud) + ' BAUD.')
        except:
            print("Failed to connect with " + str(serialPort) + ' at ' + str(serialBaud) + ' BAUD.')

    def readSerialStart(self):
        if self.thread == None:
            self.thread = Thread(target=self.backgroundThread)
            self.thread.start()

            # Block till we start receiving values
            while self.isReceiving != True:
                time.sleep(0.1)

    def getSerialData(self):
        privateData = copy.deepcopy(self.rawData[:])    # so that the 3 values in our plots will be synchronized to the same sample time
        data = privateData[0:(self.dataNumBytes)]
        value,  = struct.unpack(self.dataType, data)

        return value


    def backgroundThread(self):    # retrieve data
        time.sleep(0.1)  # give some buffer time for retrieving data
        self.serialConnection.reset_input_buffer()
        while (self.isRun):
            self.serialConnection.readinto(self.rawData)
            self.isReceiving = True
            # print(self.rawData)

    def close(self):
        self.isRun = False
        self.thread.join()
        self.serialConnection.close()
        print('Disconnected...')
        df = pd.DataFrame(self.csvData)
        df.to_csv('./data/data.csv')