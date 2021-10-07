###############################################################################
# DEVELOPED BY KANISHK                                            #############
# THIS SCRIPT DEFINES POSSIBLE DEMOS FOR ROBOT                    #############
###############################################################################

import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.getcwd()) + '/core')


###############################################################################
# MAIN SECTION                                                    #############
###############################################################################
if __name__ == '__main__':

    # Import 'Robot' class from the 'core'
    from robot import Robot

    nyu_finger = Robot()

    # To Make the robot wait for the specified seconds
    nyu_finger.wait(1)

    # Move to a Singularity Free Pose
    nyu_finger.Move_P2P(goal_pos=nyu_finger.parking_pose, mode='joint')
    nyu_finger.wait(1)

    # Moving to a target position using a target speed.
    target_pos = [np.array([0.05, -0.05, 0.1]),
                  np.array([0.0, 0.05, 0.1]),
                  np.array([0.05, 0.05, 0.1])]
    target_vel = 0.75

    print('Tracing the Circle')
    nyu_finger.Move_P2P(goal_pos=target_pos,
                        goal_vel=target_vel,
                        mode='circular')
    nyu_finger.wait(1)

    # Move to a Singularity Free Pose
    print('Moving to Park')
    nyu_finger.Move_P2P(goal_pos=nyu_finger.parking_pose, mode='joint')
    nyu_finger.wait(1)

    # Close & Exit
    nyu_finger.exit()
