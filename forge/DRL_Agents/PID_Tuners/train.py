# Library Imports
import sys
import os
from time import sleep

sys.path.append('forge')
import numpy as np

from robot_simulator import NYUFingerSimulator
from TD3 import Agent
from sklearn.metrics import mean_squared_error as mse

if __name__ == "__main__":
    # Initialize the Simulator
    package_dir = os.path.abspath(os.getcwd())
    urdf_path = package_dir + '/model/fingeredu.urdf'
    robot = NYUFingerSimulator(package_dir, urdf_path)
    q0, v0 = robot.get_state()
    robot.reset_state(q0, v0)

    # Initialize the Agent
    agent = Agent(alpha=0.001, beta=0.001, input_dims=6, tau=0.005,
                  batch_size=256, layer1_size=400, layer2_size=300, n_actions=3)
    
    # Initialize the Game and Plotters
    best_score = -np.inf
    n_games = 2500
    score_log = []
    AvgScore_log = []

    for i in range(n_games):
        score = 0
        done = False

        # Generate Random Init. & Target Pose
        reset_pose = np.random.uniform(np.pi,-np.pi,(3,))
        robot.reset_state(reset_pose, np.zeros(3))

        target_pose = np.random.uniform(np.pi,-np.pi,(3,))
        
        q_log = []
        nextq_log = []
        tau_log = []

        # Copy Variables
        q,_ = robot.get_state()
        next_q = q

        # Normal Experience Replay Tryouts
        for _ in range(250):
            
            q,_ = robot.get_state()

            assert np.array_equal(q, next_q)

            tau = agent.choose_action(np.hstack((q,target_pose))).numpy()

            # Send torque commands
            robot.send_joint_torque(tau)
            robot.step()

            # Next State
            next_q, _ = robot.get_state()

            # Store the States
            q_log.append(q)
            nextq_log.append(next_q)
            tau_log.append(tau)

            # Reward Computation
            reward = -mse(next_q, target_pose)
            score += reward

            # Done Flag Raiser
            if reward == 0.0:
                done = True

            # Normal Experience Replay
            agent.remember(np.hstack((q,target_pose)), tau, reward, np.hstack((next_q,target_pose)), done)
            agent.learn()

            if done:
                break

        # HindSight Experience Replay Tryouts
        """
        HER_target = nextq_log[-1]
        for index in range(len(q_log)):
            q = q_log[index]
            tau = tau_log[index]
            next_q = nextq_log[index]
            HER_reward = -mse(next_q, HER_target)
            agent.remember(np.hstack((q,HER_target)), tau, HER_reward, np.hstack((next_q,HER_target)), np.array_equal(next_q, HER_target))
            agent.learn()"""

        # Log game scores
        score_log.append(score)
        AvgGameScore = np.mean(score_log[-100:])
        AvgScore_log.append(AvgGameScore)
        np.save('forge/DRL_Agents/PID_Tuners/data/score_log', score_log, allow_pickle=False)
        np.save('forge/DRL_Agents/PID_Tuners/data/AvgScore_log', AvgScore_log, allow_pickle=False)

        # Store the best Model
        if AvgGameScore > best_score:
            best_score = AvgGameScore
            agent.save_models()

        print (f'Game: {i} \t ACC. Scores:{score:.1f} \t AVG. Scores: {AvgGameScore:.1f} \t Done Status: {done}')
            
            
