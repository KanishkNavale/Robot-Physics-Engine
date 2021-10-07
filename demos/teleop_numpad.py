###############################################################################
# DEVELOPED BY KANISHK                                            #############
# THIS SCRIPT DEFINES POSSIBLE DEMOS FOR ROBOT                    #############
###############################################################################

import sys
import os
sys.path.append(os.path.abspath(os.getcwd()) + '/core')


###############################################################################
# MAIN SECTION                                                    #############
###############################################################################
if __name__ == '__main__':

    # Import 'Robot' class from the 'core'
    from robot import Robot

    nyu_finger = Robot(connect_hardware=False, controller='keyboard')

    # Move to a Singularity Free Pose
    nyu_finger.Move_P2P(goal_pos=nyu_finger.parking_pose, mode='joint')
    nyu_finger.wait(1)
