###############################################################################
# DEVELOPED BY KANISHK                                            #############
# THIS SCRIPT DEFINES POSSIBLE DEMOS FOR ROBOT                    #############
###############################################################################

import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.getcwd()) + '/core')


###############################################################################
# MAIN SECTION                                                    #############
###############################################################################
if __name__ == '__main__':

    # Import 'Robot' class from the 'core' and 'memory'
    from robot import Robot

    nyu_finger = Robot()

    # To Make the robot wait for the specified seconds
    nyu_finger.wait(1)

    # Move to a Singularity Free Pose
    nyu_finger.Move_P2P(goal_pos=nyu_finger.parking_pose, mode='joint')
    nyu_finger.wait(1)

    # Move robot with target speed and position
    target_pos = np.array([-0.1, -0.1, 0.02])
    target_vel = 1
    nyu_finger.Move_P2P(goal_pos=target_pos,
                        goal_vel=target_vel,
                        mode='linear')
    nyu_finger.wait(1)

    # Move robot with target speed and position
    target_pos = np.array([-0.1, 0.1, 0.02])
    target_vel = 0.5
    nyu_finger.Move_P2P(goal_pos=target_pos,
                        goal_vel=target_vel,
                        mode='linear')
    nyu_finger.wait(1)

    # Move robot with target speed and position
    target_pos = np.array([0.1, 0.1, 0.02])
    target_vel = 0.5
    nyu_finger.Move_P2P(goal_pos=target_pos,
                        goal_vel=target_vel,
                        mode='linear')
    nyu_finger.wait(1)

    # Move robot with target speed and position
    target_pos = np.array([0.1, -0.1, 0.02])
    target_vel = 0.5
    nyu_finger.Move_P2P(goal_pos=target_pos,
                        goal_vel=target_vel,
                        mode='linear')
    nyu_finger.wait(1)

    # Move to a Singularity Free Pose
    nyu_finger.Move_P2P(goal_pos=nyu_finger.parking_pose, mode='joint')
    nyu_finger.wait(1)

    # Close & Exit
    nyu_finger.exit()
