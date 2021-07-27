import matplotlib.pyplot as plt
import numpy as np

scores     = np.load('forge/DRL_Agents/PID_Tuners/data/score_log.npy')
avg_scores = np.load('forge/DRL_Agents/PID_Tuners/data/AvgScore_log.npy')

plt.plot(scores, alpha=0.25, label='Acc. Scores')
plt.plot(avg_scores, label='Avg. Scores')
plt.legend(loc='best')
plt.grid(True)
plt.show()