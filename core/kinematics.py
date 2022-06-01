   def ForwardKinematics(self, q, v=np.zeros(3)) -> np.ndarray:
        """
        Description:
            1. Computes the Cartesian Position & Velocity of the robot
            2. The TCP position is w.r.t. the 'tip_link' of the robot.

        Args:
            q -> np.array (3,) : Joint Positions of the robot.
            v -> np.array (3,) : Joint Velocities of the robot.

        Returns:
            np.array (3,) : Cartesian Position of the robot.
            np.array (3,) : Cartesian Velocity of the robot.
        """
        index = self.EOAT_ID
        frame = pin.ReferenceFrame.LOCAL
        pin.forwardKinematics(self.model, self.model_data, q, v)
        pin.updateFramePlacements(self.model, self.model_data)
        pos = pin.updateFramePlacement(self.model, self.model_data, index)
        vel = pin.getFrameVelocity(self.model, self.model_data, index, frame)
        return np.array(pos)[:3, -1], np.array(vel)[:3]

    def FrameJacobian(self, q: np.ndarray) -> np.ndarray:
        """
        Description:
            1. Computes the Jacobian Tensor of the robot.

        Args:
            q -> np.array (3,) : Joint Positions of the robot.

        Returns:
            np.array (3,3): Jacobian Tensor of the robot.
        """
        frame = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        pin.computeJointJacobians(self.model, self.model_data, q)
        return pin.getFrameJacobian(self.model, self.model_data, self.EOAT_ID, frame)[:3]

    def FrameHessian(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Description:
            1. Computes the Time Derivative Jacobian Tensor of the robot.

        Args:
            q -> np.array (3,) : Joint Positions of the robot.
            v -> np.array (3,) : Joint Velocities of the robot.

        Returns:
            np.array (3,3): Time Derivative Jacobian Tensor of the robot.
        """
        frame = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        pin.computeJointJacobians(self.model, self.model_data, q)
        pin.framesForwardKinematics(self.model, self.model_data, q)
        pin.computeJointJacobiansTimeVariation(self.model, self.model_data, q, v)
        return pin.getJointJacobianTimeVariation(self.model, self.model_data, self.EOAT_ID, frame)[:3]
