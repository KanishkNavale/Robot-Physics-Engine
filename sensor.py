# Library Imports
from os import minor
import serial
import matplotlib.pyplot as plt
from time import sleep
from collections import deque
import numpy as np
import sys
from threading import Thread

# Init. Plotter
plt.style.use('dark_background')

class HapticSensor:
    def __init__(self, port, baudrate, mem_len, style='chart'):
        self.port = port
        self.baudrate = baudrate
        self.mem_len = mem_len
        self.style = style
    
        # Init. variables for readingline
        self.buffer = bytearray()

        # Open the port
        for i in range(10):
            print (f'Trying to open Port {self.port}')
            try:
                # Init. Sensor
                self.sensor = serial.Serial(self.port, self.baudrate)
                self.sensor.close()
                self.sensor.open()
                print (f'Successfully Opened Port: {self.port}')
                break
            except Exception as e:
                print (f'Port Opening Error')
                self.port = self.port[:-1]+str(int(self.port[-1])+1)
                print (f'Attempt to open Port: {self.port}')
                if i==9:
                    print ('Sensor Port Search and Init. Failed')
                    sys.exit(0)
                continue

        # List of Readings
        self.channels = [0,1,2,3,6,7,8,9,12,13,14,15,18,19,20,21]

        # Read and Discard Initial Readings 
        for _ in range(10):
            pre_data = self.data_formatted()

        #Create Memory
        self.memory = [deque(maxlen=self.mem_len) for _ in range(len(pre_data))]
            
        # Plotters
        self.fig, self.ax = plt.subplots(1, 1)
        plt.show(block=False)

        # Parallel Plotter
        self.plot = Thread(target=self.plotter)
        self.plot.start()
        
                    
    def read_buffer(self):
        i = self.buffer.find(b"\n")
        if i >= 0:
            r = self.buffer[:i+1]
            self.buffer = self.buffer[i+1:]
            return r
        while True:
            i = max(1, min(2048, self.sensor.in_waiting))
            data = self.sensor.read(i)
            i = data.find(b"\n")
            if i >= 0:
                r = self.buffer + data[:i+1]
                self.buffer[0:] = data[i+1:]
                return r       
            else:
                self.buffer.extend(data)

    def data_formatted(self):
            data = self.read_buffer()
            data =  str(data)
            data = np.array(data[12:-5].split())
            try:
                data = data[self.channels]
                for i in range(len(data)):
                    data[i]= int(data[i],16)
                if type(data) is np.ndarray:
                    return data
                else:
                    raise Exception
            except:
                self.data_formatted()
            
    def plotter(self):
        while(True):
            data = self.data_formatted()
            self.ax.clear()

            if type(data) is np.ndarray:
                for i in range(len(data)):
                    self.memory[i].append(data[i])

                if self.style == 'chart':
                    for i in range(len(data)):
                        plt.plot(list(self.memory[i]), label='Channel:'+str(self.channels[i]))
                    self.ax.legend(loc='center left')
                    self.ax.grid(False)

                if self.style == 'matrix':
                    data = np.reshape(data, (4,4)).astype(np.float)
                    self.ax.matshow(data)
                    print (data)
                    for x in range (data.shape[0]):
                        for y in range(data.shape[0]):
                            self.ax.text(x, y, data[y, x])
                    self.ax.set_xticks(np.arange(-0.5, data.shape[0], 1), minor=False)
                    self.ax.set_yticks(np.arange(-0.5, data.shape[0], 1), minor=False)
                    self.ax.set_xticklabels([])
                    self.ax.set_yticklabels([])
                    self.ax.grid(which='major', color='w', linestyle='-', linewidth=5)

            self.fig.canvas.draw()
                
              
if __name__ == '__main__':
    sensor = HapticSensor('/dev/ttyACM0',115200, 100, style='matrix')
    plt.show()
