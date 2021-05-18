
#///////////////////////////////////////////////////////////////////////////////
#// BSD 3-Clause License
#//
#// Copyright (C) 2020, New York University , Max Planck Gesellschaft
#// Copyright note valid unless otherwise stated in individual files.
#// All rights reserved.
#///////////////////////////////////////////////////////////////////////////////

import numpy as np
import pybullet as p


import pinocchio_bullet_wrapper
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper

import time
import os.path

class NYUFingerSimulator:
    def __init__(self):
        # Connet to pybullet and setup simulation parameters.
        p.connect(p.GUI)
        p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(fixedTimeStep=1.0/1000.0, numSubSteps=1)

        # Zoom onto the robot.
        p.resetDebugVisualizerCamera(1.0, 50, -35, (0., 0., 0.))

        # Disable the gui controller as we don't use them.
#         p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
        
        ###
        # Load the finger robot
        robotStartPos = [0.,0,.0]
        robotStartOrientation = p.getQuaternionFromEuler([0,0,0])

        urdf_path = '../urdf/fingeredu.urdf'

        self.robotId = p.loadURDF(urdf_path, robotStartPos,
                robotStartOrientation, flags=p.URDF_USE_INERTIA_FROM_FILE,
                useFixedBase=True)
        
        # Create pinocchio robot
        self.pin_robot = RobotWrapper.BuildFromURDF(urdf_path, [os.path.abspath('../urdf/')])
        
        # Create a wrapper between the pinocchio and pybullet robot.
        joint_names = [
        #     'base_to_finger',
            'finger_base_to_upper_joint', # HAA
            'finger_upper_to_middle_joint', # HFE
            'finger_middle_to_lower_joint', # KFE
        #     'finger_lower_to_tip_joint'
        ]

        self.robot = pinocchio_bullet_wrapper.PinBulletWrapper(self.robotId, self.pin_robot, joint_names, [], True)
        
    def get_state(self):
        return self.robot.get_state()
    
    def reset_state(self, q, v):
        self.robot.reset_state(q, v)
        
    def send_joint_torque(self, tau):
        self.robot.send_joint_command(tau)
        
    def step(self):
        time.sleep(0.001)
        p.stepSimulation()