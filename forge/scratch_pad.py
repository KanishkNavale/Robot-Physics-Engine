# Library Imports
from threading import Thread
import matplotlib.pyplot as plt
from time import sleep
import matplotlib.animation as animation
from matplotlib import style
from collections import deque
import numpy as np
plt.style.use('dark_background')


class Main:
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1,1,1)      
        self.data = deque(maxlen=50)
         
        self.animator = Thread(target=self.read_sensor)
        self.animator.start()
        
        
    def read_sensor(self):
        while True:
            self.ax.clear()   
            self.data.append(np.random.uniform(1, -1))
            plt.plot(self.data, label='data')
            self.ax.grid(True)
            self.ax.legend(loc='center left')
            self.fig.canvas.draw()
        
if __name__ == "__main__":
    plotter = Main()
    plt.show()  
    
