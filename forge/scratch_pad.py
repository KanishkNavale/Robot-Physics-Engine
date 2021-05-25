def set_JointStates(self, target_q):
        # Goal Set Points
        q_goal = target_q

        # For the Ki Tuning
        Err_log=[]
        Err_log.append(q_goal - q)
        
        for i in range(1, N+1):
            t = i * tau
            
            # Compute Step Angles
            q = q_goal - (q*t)/T
            v = v_goal - (v*t)/T
            a = a_goal - (a*t)/T

            # PID controller
            u = self.Kp*(q_goal - q) + self.Ki* np.sum(Err_log) + self.Kd*(v_goal - v) 

            # Compute Other Inverse Dynamics Compensations
            cu = pin.rnea(self.model, self.model_data, q, v, a)

            # Send torque commands
            self.device.send_joint_torque(u + cu)
            print (q_goal, q)

            q, v = self.device.get_state()
            self.viz.display(q)

            Err_log.append(q_goal - q)
            sleep(tau)
            
        # Restart Dynamic Compensation
        print ('Restarting Dynamic Compensation')
        self.SetDynamicCompensation = Thread(target=self.DynamicCompensation, args=(q, v, np.zeros(3)))
        self.SDC_Flag.clear()
        self.SetDynamicCompensation.start()