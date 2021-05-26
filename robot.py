######################################################################################################################
# DEVELOPED BY KANISHK                                                                                   #############
# MAX PLANCK INSTITUTE FOR INTELLIGENT SYSTEMS                                                           #############
# STUTTGART, GERMANY                                                                                     #############
                                                                                                         #############
# THIS SCRIPT CONSTRUCTS A ROBOT CLASS BASED ON FUNCTIONALITIES FROM THE 'FORGE'                         #############
######################################################################################################################

# Library Imports
import threading
from nyu_finger import NYUFingerReal

import pinocchio as pin
from pinocchio.visualize import GepettoVisualizer

import numpy as np
np.set_printoptions(precision=4, suppress=True)

import matplotlib.pyplot as plt
plt.style.use('dark_background')
import matplotlib.animation as animation

import os
from threading import Thread
from time import sleep

import serial
from collections import deque

#######################################################################################################################
# Class Construct of the robot                                                                            #############
#######################################################################################################################
class Robot:
    """ Contructs a class for the 'nyu_finger' robot """

    def __init__(self, device, comms):
        # Init. Hardware
        self.device = device
        try:
            self.device.initialize(comms)
        except:
            print('Something Went Wrong!')

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
        self.viz = GepettoVisualizer(self.model, self.collision_model, self.visual_model)
        self.viz.initViewer()
        self.viz.loadViewerModel("nyu_finger")
        self.viz.displayCollisions(True)
        self.viz.displayVisuals(True)

        # Get the FrameID of the EOAT
        self.EOAT_ID = self.model.getFrameId('finger_tip_link')
        # Get the FrameID of the Base
        self.Base_ID = self.model.getFrameId('base_link')

        # Offset Joint Values
        q, v = self.device.get_state()
        self.joint_offsets = q
        self.vel_offsets = v

        # Variables for Dynamic Compensation
        q, v = self.calibrated_states()
        self.set_pos = q
        self.set_vel = v
        
        # Enable Dynamic Compensation
        self.SetDynamicCompensation = Thread(target=self.DynamicCompensation)
        self.SDC_Flag = threading.Event()
        self.SDC_Flag.clear()
        self.SetDynamicCompensation.start()

        # Refresh Display
        self.update_viz = Thread(target=self.refresh_viz)
        self.update_viz.start()

        # Init. the Haptic Sensor
        try: 
            self.sensor_port = serial.Serial('/dev/ttyACM0',115200)
            self.dataframe = self.ReadLine(self.sensor_port)
        except Exception as e:
            print (f'Sensor Error: {e}')
        
        try:       
            self.sensor_port.close()
            self.sensor_port.open()
            
            # Discard firstline of the reading and discard it!
            self.dataframe.readline()
            self.channels = [0,1,2,3,6,7,8,9,12,13,14,15,18,19,20,21]
            self.readings = [deque(maxlen=100) for i in range(self.channels[-1])]
            
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)
            self.plot_readings = Thread(target=self.read_sensor)
            self.plot_readings.start()
        except Exception as e:
            print (f'Sensor Error: {e}')  
            

    ####################################################################################################################
    # Refresh Display                                                                                     ##############
    ####################################################################################################################
    def refresh_viz(self):
        refresh_rate = 60 #(60FPS)
        while True:
            q, _ = self.calibrated_states()
            self.viz.display(q)
            sleep(1/refresh_rate)

    ####################################################################################################################
    # Read Haptic Sensor                                                                                  ##############
    ####################################################################################################################
    # Class to a extract bytearray
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

    # Definition to add extract data from bytearray
    def read_sensor(self):
        while True:
            data = self.dataframe.readline()
            data =  str(data)
            data = data[12:-5].split()
            
            if len(data)== 36:
                self.ax.clear()

                for i in self.channels:
                    self.readings[i].append(int(data[i], 16))
                    plt.plot(list(self.readings[i]), label=str(i))
                    
                self.ax.grid(True)
                self.ax.legend(loc='center left')
                self.fig.canvas.draw()

    ####################################################################################################################
    # KINEMATICS                                                                                          ##############
    ####################################################################################################################
    # Compute Joint Offsets
    def calibrated_states(self):
        q, v = self.device.get_state()
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
    def P2P_Planner(self, init_pos, goal_pos, interpolation='joint', track_EOAT=True):
        # Initiate Callers
        q = init_pos
        y_target = goal_pos.copy()
        init_y = self.compute_ForwardKinematics(q)

        # Constants
        W = 1e-4 * np.eye(3)

        # Track EOAT
        if track_EOAT:
            EOAT = []
            EOAT.append(init_y)

        # HyperParams-> Needs to be tuned for large P2P Motion
        if interpolation=='joint':
            max_steps = 100
            smooth = 0.1
        if interpolation=='linear':
            max_steps = 1000
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
            self.set_JointStates(q)

            if track_EOAT:
                EOAT.append(y)

        if track_EOAT:
            return q, np.vstack(EOAT)
        else:
            return q
    

    # Point based Trajectory Planner
    def Trajectory_Planner(self, joint_pos, Point_List, track_EOAT=True):
        q = joint_pos
        # Track EOAT
        if track_EOAT:
            EOAT = []
        # Invoke P2P_Planner for each Point
        for i, point in enumerate(Point_List):
            q, trace_points = self.P2P_Planner(q, point)
            EOAT.append(trace_points)
        if track_EOAT:
            return np.vstack(EOAT)


    # Plot of EOAT Position 
    def plot_EOATPosition(self, trace_points, save=False):
        ax = plt.axes(projection='3d')
        ax.plot3D(trace_points[:,0],trace_points[:,1],trace_points[:,2], c='red', label='Trajectory')
        ax.legend(loc='best')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
        if save:
           plt.savefig('Trajectory_Plot.png') 


    ####################################################################################################################
    # COLLISIONS COMPUTATION                                                                              ##############
    ####################################################################################################################
    # Definition to Compute Collisions
    def compute_Collisions(self,q):
        pin.computeCollisions(self.model, self.model_data, self.collision_model ,self.collision_data, q, False)
        for k in range(len(self.collision_model.collisionPairs)): 
            cr = self.collision_data.collisionResults[k]
        self.viz.displayCollisions(True)
        self.viz.displayVisuals(True)
        return cr.isCollision()


    ####################################################################################################################
    # JOINT DYNAMICS                                                                                      ##############
    ####################################################################################################################
    # Definition to Set Joint Torques using dynamics
    def set_JointStates(self, target_q):
        # Disable Dynamic Compensation
        if self.SetDynamicCompensation.is_alive():
            print ('Terminating Dynamic Compensation!')
            self.SDC_Flag.set()
            self.SetDynamicCompensation.join()
        
        tau = 0.01
        N = 100
        T = N*tau

        # Goal Set Points
        q_goal = target_q
        v_goal = np.zeros(3)
        a_goal = np.zeros(3)

        # Init. Values
        q, v = self.calibrated_states()
        a = a_goal.copy()

        # Joint Dynamics Defaults
        self.Kp = 1
        self.Kd = 2 * np.sqrt(self.Kp)
        self.Ki = 1e-10

        # For the Ki Tuning
        Err_log=[]
        Err_log.append(q_goal - q)
        
        for i in range(1, N+1):
            t = i * tau
            
            # Compute Step Angles
            q += (t/T) * (q_goal - q)
            v += (t/T) * (v_goal - v)
            a += (t/T) * (a_goal - a)
            
            # PID controller
            u  = self.Kp*(q_goal - q)
            u += self.Ki* np.sum(Err_log) 
            u += self.Kd*(v_goal - v) 

            # Compute Gravity Compensation
            gu = pin.computeGeneralizedGravity(self.model, self.model_data, q)

            # Send torque commands
            self.device.send_joint_torque(u + gu)

            q, v = self.calibrated_states()
            self.set_pos = q
            self.set_vel = v
            print (q_goal, q)

            Err_log.append(q_goal - q)
            sleep(tau)
            
        # Restart Dynamic Compensation
        print ('Restarting Dynamic Compensation')
        self.SetDynamicCompensation = Thread(target=self.DynamicCompensation)
        self.SDC_Flag.clear()
        self.SetDynamicCompensation.start()


    ####################################################################################################################
    # GRAVITY COMPENSATION + Other Inverse Dynamics Compensations                                         ##############
    ####################################################################################################################
    # Definition to compute Inverse Dynamics Gravity + Inertia + NLE Forces + Centrifugal & Corrolis 
    def DynamicCompensation(self):
        print ('Setting in Dynamic Compensation!')
        while not self.SDC_Flag.is_set():
            # Computing Inverse Dynamics for hold torque
            gu = pin.computeGeneralizedGravity(self.model, self.model_data, self.set_pos)
            tau = pin.rnea(self.model, self.model_data, self.set_pos, self.set_vel, np.zeros(3))
            self.device.send_joint_torque(tau)
  
 
#######################################################################################################################
# MAIN Runtime                                                                                           ##############
#######################################################################################################################
if __name__ == "__main__":
    robot = Robot(NYUFingerReal(), 'enp5s0')
    #robot.set_JointStates(np.array([0, 0, np.pi/4]))
    plt.show()
        
