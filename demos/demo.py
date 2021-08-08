######################################################################################################################
# DEVELOPED BY KANISHK                                                                                   #############                                                                                    #############
                                                                                                         #############
# THIS SCRIPT DEFINES POSSIBLE DEMOS FOR ROBOT                                                           #############
######################################################################################################################

# Import 'Robot' class from the 'core'
import sys
sys.path.append('core')
from robot import Robot

import numpy as np


######################################################################################################################
# MAIN SECTION                                                                                           #############
######################################################################################################################
if __name__ == '__main__':
    nyu_finger = Robot()
    
    # To Make the robot wait for the specified seconds
    nyu_finger.wait(1)

    # Moving from One point to Another
    ## In mode='joint' the target position is given as 'Joint Angles'
    print('Executing Joint Movements')
    nyu_finger.Move_P2P(np.array([0, np.pi/4, -np.pi/4]), mode='joint')
    nyu_finger.wait(1)
    nyu_finger.Move_P2P(np.array([0, -np.pi/4, np.pi/4]), mode='joint')
    nyu_finger.wait(1)

    ## In mode='linear' the target position is given as 'Points in Cartesian Space'
    print('Executing Linear Movements')
    nyu_finger.Move_P2P(np.zeros(3), mode='linear')
    nyu_finger.wait(1)
    nyu_finger.Move_P2P(np.array([0, 0, 0.1]), mode='linear')
    nyu_finger.wait(1)

    ## In mode='circular' the target position is given as 'Points in Cartesian Space'
    ## In addition use the 'C' Arg. to specify two points. The circle generated fits these two points
    ## Initiate 'goal_pos' Arg. as None 
    print('Executing Circular Movements')
    nyu_finger.Move_P2P(None, mode='circular', C=[np.array([-0.2, -0.1, 0.15]), np.array([0.1, 0.2, 0.1]), np.array([0.1, -0.1, 0.1])])
    nyu_finger.wait(1)

    ## The Robot can move to several points using the below function
    ## Pass a list of Points in Cartesian Space as Args.
    print ('Executing Free Path Movements')
    nyu_finger.Trajectory_Planner([np.zeros(3), np.array([0.1, 0, 0.1]), np.array([0, 0.1, 0.1]), np.zeros(3)])
    nyu_finger.wait(1)
    
    ## The path can be visualized as graph too
    trace = nyu_finger.Move_P2P(None, mode='circular', C=[np.array([-0.1, -0.1, 0.1]), np.array([0.1, 0.1, 0.1]), np.array([-0.1, 0.1, 0.05])])
    nyu_finger.plot_TracePath(trace)
    