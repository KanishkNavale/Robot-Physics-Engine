from time import sleep
import numpy as np

# Goal Set Points
q_goal = np.random.uniform(-1,1,(6,))

q = np.random.uniform(-1,1,(6,))

Err_log=[]
Err_log.append(q_goal - q)
tau = 0.1
N = 5
T = tau*N

for i in range(1, N+1):
    t = i * tau
    
    # Compute Step Angles
    q += (i/N)*(q_goal - q)

    Err_log.append(q_goal - q)
    print(np.around(q_goal - q, 4))
            