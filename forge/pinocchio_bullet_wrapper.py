#///////////////////////////////////////////////////////////////////////////////
#// BSD 3-Clause License
#//
#// Copyright (C) 2018-2019, New York University , Max Planck Gesellschaft
#// Copyright note valid unless otherwise stated in individual files.
#// All rights reserved.
#///////////////////////////////////////////////////////////////////////////////

import pybullet as p
import pinocchio as se3
import numpy as np
from time import sleep

from pinocchio.utils import zero

class PinBulletWrapper(object):
    def __init__(self, robot_id, pinocchio_robot, joint_names, endeff_names, useFixedBase=False):
        self.nq = pinocchio_robot.nq
        self.nv = pinocchio_robot.nv
        self.nj = len(joint_names)
        self.nf = len(endeff_names)
        self.robot_id = robot_id
        self.pinocchio_robot = pinocchio_robot
        self.useFixedBase = useFixedBase

        self.joint_names = joint_names
        self.endeff_names = endeff_names

        bullet_joint_map = {}
        for ji in range(p.getNumJoints(robot_id)):
            bullet_joint_map[p.getJointInfo(robot_id, ji)[1].decode('UTF-8')] = ji

        self.bullet_joint_ids = np.array([bullet_joint_map[name] for name in joint_names])
        self.pinocchio_joint_ids = np.array([pinocchio_robot.model.getJointId(name) for name in joint_names])

        self.pin2bullet_joint_only_array = []

        if not self.useFixedBase:
            for i in range(2, self.nj + 2):
                self.pin2bullet_joint_only_array.append(np.where(self.pinocchio_joint_ids == i)[0][0])
        else:
            for i in range(1, self.nj + 1):
                self.pin2bullet_joint_only_array.append(np.where(self.pinocchio_joint_ids == i)[0][0])


        # Disable the velocity control on the joints as we use torque control.
        p.setJointMotorControlArray(robot_id, self.bullet_joint_ids, p.VELOCITY_CONTROL, forces=np.zeros(self.nj))

        # In pybullet, the contact wrench is measured at a joint. In our case
        # the joint is fixed joint. Pinocchio doesn't add fixed joints into the joint
        # list. Therefore, the computation is done wrt to the frame of the fixed joint.
        self.bullet_endeff_ids = [bullet_joint_map[name] for name in endeff_names]
        self.pinocchio_endeff_ids = [pinocchio_robot.model.getFrameId(name) for name in endeff_names]

    def _action(self, pos, rot):
        res = np.zeros((6, 6))
        res[:3, :3] = rot
        res[3:, 3:] = rot
        res[3:, :3] = se3.utils.skew(np.array(pos)).dot(rot)
        return res

    def get_force(self):
        """ Returns the force readings as well as the set of active contacts """
        active_contacts_frame_ids = []
        contact_forces = []

        # Get the contact model using the p.getContactPoints() api.
        def sign(x):
            if x >= 0:
                return 1.
            else:
                return -1.

        cp = p.getContactPoints()

        for ci in reversed(cp):
            contact_normal = ci[7]
            normal_force = ci[9]
            lateral_friction_direction_1 = ci[11]
            lateral_friction_force_1 = ci[10]
            lateral_friction_direction_2 = ci[13]
            lateral_friction_force_2 = ci[12]

            if ci[3] in self.bullet_endeff_ids:
                i = np.where(np.array(self.bullet_endeff_ids) == ci[3])[0][0]
            elif ci[4] in self.bullet_endeff_ids:
                i = np.where(np.array(self.bullet_endeff_ids) == ci[4])[0][0]
            else:
                continue

            if self.pinocchio_endeff_ids[i] in active_contacts_frame_ids:
                continue

            active_contacts_frame_ids.append(self.pinocchio_endeff_ids[i])
            force = np.zeros(6)

            force[:3] = normal_force * np.array(contact_normal) + \
                        lateral_friction_force_1 * np.array(lateral_friction_direction_1) + \
                        lateral_friction_force_2 * np.array(lateral_friction_direction_2)

            contact_forces.append(force)

        return active_contacts_frame_ids[::-1], contact_forces[::-1]

    def get_base_velocity_world(self):
        """ Returns the velocity of the base in the world frame.
        Returns:
            np.array((6,1)) with the translation and angular velocity
        """
        vel, orn = p.getBaseVelocity(self.robot_id)
        return np.array(vel + orn).reshape(6, 1)

    def get_state(self):
        # Returns a pinocchio like representation of the q, dq matrixes
        q = zero(self.nq)
        dq = zero(self.nv)

        if not self.useFixedBase:
            pos, orn = p.getBasePositionAndOrientation(self.robot_id)
            q[:3] = pos
            q[3:7] = orn

            vel, orn = p.getBaseVelocity(self.robot_id)
            dq[:3] = vel
            dq[3:6] = orn

            # Pinocchio assumes the base velocity to be in the body frame -> rotate.
            rot = np.array(p.getMatrixFromQuaternion(q[3:7])).reshape((3, 3))
            dq[0:3] = rot.T.dot(dq[0:3])
            dq[3:6] = rot.T.dot(dq[3:6])

        # Query the joint readings.
        joint_states = p.getJointStates(self.robot_id, self.bullet_joint_ids)

        if not self.useFixedBase:
            for i in range(self.nj):
                q[5 + self.pinocchio_joint_ids[i]] = joint_states[i][0]
                dq[4 + self.pinocchio_joint_ids[i]] = joint_states[i][1]
        else:
            for i in range(self.nj):
                q[self.pinocchio_joint_ids[i] - 1] = joint_states[i][0]
                dq[self.pinocchio_joint_ids[i] - 1] = joint_states[i][1]

        return q, dq

    def update_pinocchio(self, q, dq):
        """Updates the pinocchio robot.
        This includes updating:
        - kinematics
        - joint and frame jacobian
        - centroidal momentum
        Args:
          q: Pinocchio generalized position vect.
          dq: Pinocchio generalize velocity vector.
        """
        self.pinocchio_robot.computeJointJacobians(q)
        self.pinocchio_robot.framesForwardKinematics(q)
        self.pinocchio_robot.centroidalMomentum(q, dq)

    def get_state_update_pinocchio(self):
        """Get state from pybullet and update pinocchio robot internals.
        This gets the state from the pybullet simulator and forwards
        the kinematics, jacobians, centroidal moments on the pinocchio robot
        (see forward_pinocchio for details on computed quantities). """
        q, dq = self.get_state()
        self.update_pinocchio(q, dq)
        return q, dq

    def reset_state(self, q, dq):
        vec2list = lambda m: np.array(m.T).reshape(-1).tolist()

        if not self.useFixedBase:
            p.resetBasePositionAndOrientation(self.robot_id, vec2list(q[:3]), vec2list(q[3:7]))

            # Pybullet assumes the base velocity to be aligned with the world frame.
            rot = np.array(p.getMatrixFromQuaternion(q[3:7])).reshape((3, 3))
            p.resetBaseVelocity(self.robot_id, vec2list(rot.dot(dq[:3])), vec2list(rot.dot(dq[3:6])))

            for i, bullet_joint_id in enumerate(self.bullet_joint_ids):
                p.resetJointState(self.robot_id, bullet_joint_id,
                    q[5 + self.pinocchio_joint_ids[i]],
                    dq[4 + self.pinocchio_joint_ids[i]])
        else:
            for i, bullet_joint_id in enumerate(self.bullet_joint_ids):
                p.resetJointState(self.robot_id, bullet_joint_id,
                    q[self.pinocchio_joint_ids[i] - 1],
                    dq[self.pinocchio_joint_ids[i] - 1])


    def send_joint_command(self, tau):
        # TODO: Apply the torques on the base towards the simulator as well.
        if not self.useFixedBase:
            assert(tau.shape[0] == self.nv - 6)
        else:
            assert(tau.shape[0] == self.nv)

        zeroGains = tau.shape[0] * (0.,)

        p.setJointMotorControlArray(self.robot_id, self.bullet_joint_ids, p.TORQUE_CONTROL,
                forces=tau[self.pin2bullet_joint_only_array],
                positionGains=zeroGains, velocityGains=zeroGains)