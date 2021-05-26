# Library Imports
import serial
import matplotlib.pyplot as plt
from time import sleep
import matplotlib.animation as animation
from matplotlib import style
from collections import deque
import numpy as np

# Init. Plotter
plt.style.use('dark_background')
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

# Class to Readline
class ReadLine:
    def __init__(self, s):
        self.buf = bytearray()
        self.s = s

    def readline(self):
        i = self.buf.find(b"\n")
        if i >= 0:
            r = self.buf[:i+1]
            self.buf = self.buf[i+1:]
            return r
        while True:
            i = max(1, min(2048, self.s.in_waiting))
            data = self.s.read(i)
            i = data.find(b"\n")
            if i >= 0:
                r = self.buf + data[:i+1]
                self.buf[0:] = data[i+1:]
                return r
            else:
                self.buf.extend(data)

# Init. the serial port 
ser = serial.Serial('/dev/ttyACM0',115200)
dataframe = ReadLine(ser)



# Open the port
try:
    ser.close()
    ser.open()
except Exception:
    print (Exception)

# Loose the first reading
data = dataframe.readline()

# List of Readings
channels = [0,1,2,3,6,7,8,9,12,13,14,15,18,19,20,21]

# Read the readings 
ranges = [deque(maxlen=100) for i in range(36)]

def read_sensor(i):
    data = dataframe.readline()
    data =  str(data)
    data = data[12:-5].split()
    
    if len(data)== 36:
        print (data)
        for i,point in enumerate(data):
            ranges[i].append(int(point, 16))
        
        ax.clear()
        for i in channels:
            ax.plot(ranges[i], label=str(i))
        ax.grid(True)
        ax.legend(loc='center left')

    else:
        pass

ani = animation.FuncAnimation(fig, read_sensor, interval=0.1)
plt.show()
