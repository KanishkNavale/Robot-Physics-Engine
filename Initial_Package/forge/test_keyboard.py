from time import sleep
import pygame
from pygame.locals import *
import sys
import os
from robot_simulator import NYUFingerSimulator
from threading import Thread
import pinocchio as pin
import numpy as np

class Robot:
    """ Contructs a class for the 'nyu_finger' robot """

    def __init__(self):

        # Init. Simulation Env. using .urdf file
        # Load the .urdf files
        self.package_dir = os.path.abspath(os.getcwd())
        self.urdf_path = self.package_dir + '/model/fingeredu.urdf'

        # Create Models from the .urdf files
        self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(self.urdf_path, self.package_dir)
        self.collision_model.addAllCollisionPairs()

        # Create datas from models
        self.model_data = self.model.createData()
        self.collision_model.addAllCollisionPairs()
        self.collision_data = pin.GeometryData(self.collision_model)

        # Display the models in viewer
        """ Using the Simulator to display the model """
        self.simulator = NYUFingerSimulator(self.package_dir, self.urdf_path)
            
        # Get the FrameID of the EOAT
        self.EOAT_ID = self.model.getFrameId('finger_tip_link')
        # Get the FrameID of the Base
        self.Base_ID = self.model.getFrameId('base_link')

        # Offset Joint Values
        q, v = self.simulator.get_state()
        self.joint_offsets = q
        self.vel_offsets = v

        # Variables for Dynamic Compensation
        q, v = self.calibrated_states()
        self.set_pos = q
        self.set_vel = v

        # Refresh Display
        self.update_simulator = Thread(target=self.refresh_viz)
        self.update_simulator.start()

        # Use Keyboard as a controller
        pygame.init()
        pygame.display.set_mode((300, 300))
        self.controller = Thread(target=self.controller)
        self.controller.start()

        # Initial Position
        self.pose = self.compute_ForwardKinematics(self.set_pos)
        print (self.pose)

    ####################################################################################################################
    # Refresh Display                                                                                     ##############
    ####################################################################################################################
    def refresh_viz(self):
        refresh_rate = 60 #(60FPS)
        while True:
            q, v = self.calibrated_states()
            self.simulator.reset_state(q, v)
            sleep(1/refresh_rate)


    ####################################################################################################################
    # Keyboard Listener                                                                                   ##############
    ####################################################################################################################
    def controller(self):
        while 1:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_KP6:
                        print (f'Moving Robot: Direction +Y')
                        self.pose = self.pose + np.array([0, 0.005, 0])
                        self.P2P_Planner(self.pose)
                    if event.key == pygame.K_KP4:
                        print (f'Moving Robot: Direction -Y')
                        self.pose = self.pose + np.array([0, -0.005, 0])
                        self.P2P_Planner(self.pose)
                    if event.key == pygame.K_KP8:
                        print (f'Moving Robot: Direction -X')
                        self.pose = self.pose + np.array([-0.005, 0.0, 0])
                        self.P2P_Planner(self.pose)
                    if event.key == pygame.K_KP2:
                        print (f'Moving Robot: Direction +X')
                        self.pose = self.pose + np.array([0.005, 0.0, 0])
                        self.P2P_Planner(self.pose)
                    if event.key == pygame.K_UP:
                        print (f'Moving Robot: Direction +Z')
                        self.pose = self.pose + np.array([0.0, 0.0, 0.005])
                        self.P2P_Planner(self.pose)
                    if event.key == pygame.K_DOWN:
                        print (f'Moving Robot: Direction -Z')
                        self.pose = self.pose + np.array([0.0, 0.0, -0.005])
                        self.P2P_Planner(self.pose)
            sleep(0.05)


    ####################################################################################################################
    # KINEMATICS                                                                                          ##############
    ####################################################################################################################
    # Compute Joint Offsets
    def calibrated_states(self):
        q, v = self.simulator.get_state()
        return q - self.joint_offsets, v - self.vel_offsets


    # Computing Forward Kinematics
    def compute_ForwardKinematics(self, q):
        pin.framesForwardKinematics(self.model, self.model_data, q)
        pos = np.array(self.model_data.oMf[self.EOAT_ID].translation)
        return pos


    # Compute the Jacobian
    def compute_Jacobian(self, q):
        pin.computeJointJacobians(self.model, self.model_data, q)
        return pin.getFrameJacobian(self.model, self.model_data, self.EOAT_ID, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3]


    # Trajectory Planning for Point-to-Point(P2P) Motion
    def P2P_Planner(self, goal_pos, interpolation='linear', track_EOAT=True):
        # Initiate Callers
        q, _ = self.calibrated_states()
        y_target = goal_pos.copy()
        init_y = self.compute_ForwardKinematics(q)

        # Constants
        W = 1e-4 * np.eye(3)

        # Track EOAT
        if track_EOAT:
            EOAT = []
            EOAT.append(init_y)

        # HyperParams-> Needs to be tuned for large NYUFingerSimulator()P2P Motion
        if interpolation=='joint':
            max_steps = 10
            smooth = 0.5
        if interpolation=='linear':
            max_steps = 100
            smooth = 0.75

        for i in range(1, max_steps):
            y = self.compute_ForwardKinematics(q)
            J = self.compute_Jacobian(q) 
            if interpolation=='joint':
                pass
            if interpolation=='linear':
                y_target = init_y + (i/max_steps)*(goal_pos-y)

            # Compute Step Angle
            q += smooth * np.linalg.inv(J.T @ J + W) @ J.T @ (y_target - y)
            self.simulator.reset_state(q, np.zeros(3))

            if track_EOAT:
                EOAT.append(y)

        if track_EOAT:
            return q, np.vstack(EOAT)
        else:
            return q

if __name__ == '__main__':
    robot = Robot()
        