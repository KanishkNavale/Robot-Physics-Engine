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
ser = serial.Serial('/dev/ttyACM2',115200)
dataframe = ReadLine(ser)



# Open the port
try:
    ser.close()
    ser.open()
except Exception:
    print (Exception)

# Loose the first reading
data = dataframe.readline()

while True:
    data = dataframe.readline()
    data =  str(data)
    print()
    sleep(1)
"""
# Read the readings 
x1 = deque(maxlen=50)
x2 = deque(maxlen=50)
x3 = deque(maxlen=50)
x4 = deque(maxlen=50)
x5 = deque(maxlen=50)
x6 = deque(maxlen=50)
x7 = deque(maxlen=50)
x8 = deque(maxlen=50)


def read_sensor(i):
    try:
        data = dataframe.readline()
        data =  str(data)
        
        data = data[12:52].split()
        x1 = int(data[0], 16)
        x2 = int(data[1], 16)
        x3 = int(data[2], 16)
        x4 = int(data[3], 16)
        x5 = int(data[4], 16)
        x6 = int(data[5], 16)
        x7 = int(data[6], 16)
        x8 = int(data[7], 16)
        matrix = np.matrix([x2, x1],[x4, x3],[x6, x5],[x8 ,x7])
        ax.clear()
        ax.imshow(matrix, aspect= 'auto')
        #ax.grid(True)

    except:
        print (Exception)

ani = animation.FuncAnimation(fig, read_sensor, interval=1)
plt.show()
"""



