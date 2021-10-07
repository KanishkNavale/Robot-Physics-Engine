###############################################################################
# DEVELOPED BY KANISHK                                            #############
# STUTTGART, GERMANY                                              #############
###############################################################################

import sys
import os
import numpy as np
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

    # Moving to a target position using default speed.
    target_pos = np.array([0, 0, 0])
    nyu_finger.Move_P2P(goal_pos=target_pos, mode='joint')
    nyu_finger.wait(1)

    # Moving to a target position using a target speed.
    target_pos = np.array([0, 0, -np.pi / 2])
    target_vel = np.deg2rad(90)
    nyu_finger.Move_P2P(goal_pos=target_pos, goal_vel=target_vel, mode='joint')
    nyu_finger.wait(1)

    # Moving to a target position using a target speed.
    target_pos = np.array([0, 0, np.pi / 2])
    target_vel = np.deg2rad(180)
    nyu_finger.Move_P2P(goal_pos=target_pos, goal_vel=target_vel, mode='joint')
    nyu_finger.wait(1)

    # Moving to a target position using a target speed.
    target_pos = np.array([np.pi / 4, np.pi / 4, -np.pi / 4])
    target_vel = np.deg2rad(360)
    nyu_finger.Move_P2P(goal_pos=target_pos, goal_vel=target_vel, mode='joint')
    nyu_finger.wait(1)

    # Moving the robot in parking position.
    target_pos = nyu_finger.parking_pose
    nyu_finger.Move_P2P(goal_pos=target_pos, mode='joint')
    nyu_finger.wait(1)

    # Close & Exit
    nyu_finger.exit()
