############################################################################
"""
Testing Multiprocess and Multithreading for running dynamics in background! 
"""
############################################################################

from multiprocessing import Process
from threading import Thread
import threading
from time import sleep

class check_multiprocess:
    def __init__(self) -> None:
        self.loopHandler = Process(target=self.loop)
        self.setLoopHandler = False
        
    def loop(self):
        while self.setLoopHandler:
            if not self.setLoopHandler:
                print ('RECEIVED!')
            print ('loop')
            sleep(1)

    def run(self):
        print ("\n Testing MultiProcess Function \n")
        print('Starting loop')
        self.setLoopHandler = True
        self.loopHandler.start()
        sleep(3)
        
        print('Terminating loop')
        self.setLoopHandler = False
        self.loopHandler.terminate()
        self.loopHandler.join()
        print('Loop Terminated')
        
        print('Starting loop Again')
        self.setLoopHandler = True
        self.loopHandler = Process(target=self.loop)
        self.loopHandler.start()
        sleep(3)
        
        print('Terminating loop')
        self.setLoopHandler = False
        self.loopHandler.terminate()
        self.loopHandler.join()
        print('Loop Terminated')



class check_threading:
    def __init__(self) -> None:
        self.loopHandler = Thread(target=self.loop)
        self.event = threading.Event()
        self.event.clear()

    def loop(self):
        while not self.event.is_set():
            print ('loop')
            sleep(1)

    def run(self):
        print ("\n Testing MultiThreading Function \n")
        print('Starting loop')
        self.loopHandler.start()
        sleep(3)


        print('Terminating loop')
        self.event.set()
        self.loopHandler.join()
        print('Loop Terminated')

        
        print('Starting loop Again')
        self.event.clear()
        self.loopHandler = Thread(target=self.loop)
        self.loopHandler.start()
        sleep(3)

        print('Terminating loop')
        self.event.set()
        self.loopHandler.join()
        print('Loop Terminated')

if __name__ == "__main__":
    CM = check_multiprocess()
    CM.run()
    CT = check_threading()
    CT.run()


################################################################################
"""
RESULTS,
Multiprocess responds well than Multithreading but, terminates ethernet comms.
"""
################################################################################