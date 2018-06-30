import pymc3 as pm
import glob
import pandas as pd
import numpy as np
import theano
import theano.tensor as T

import seaborn as sns
import matplotlib.pyplot as plt


# Step 1: Define update_Q in Theano scan to avoid the loop over trials
from update_Q import update_Q, p_from_Q

# Get agent data (choices, rewards)
n_trials = 5
alpha = 0.5
beta = 3

agent_data = pd.read_csv(glob.glob('C:/Users/maria/MEGAsync/SLCN/PSGenRec/*.csv')[0])
left_choices = 1 - np.array(agent_data['selected_box'])[:n_trials]
right_choices = np.array(agent_data['selected_box'])[:n_trials]
rewards = agent_data['reward'].tolist()[:n_trials]

Q_left = update_Q(rewards, left_choices, alpha).T[0]
Q_right = update_Q(rewards, right_choices, alpha).T[0]

p_left = p_from_Q(Q_left, Q_right, beta)
p_right = p_from_Q(Q_right, Q_left, beta)

print(np.column_stack((Q_left, Q_right, p_left, p_right, rewards, left_choices)))



#
# with pm.Model() as model:
#
#     # Specify parameters
#     alpha = pm.Uniform('alpha', lower=0, upper=1)
#     beta = pm.Normal('beta', mu=3, sd=2)
#
#     # Specify agent
#     pchoice = 0.5 * np.ones(2)
#     Q = 0.5 * np.zeros(2)
#
#     # Run the task
#
#
#
#     for trial in range(len(choice)):
#
#         pchoice[0] = 1 / (1 + np.exp(beta * (Q[1] - Q[0])))
#         pchoice[1] = 1 - pchoice[0]
#
#         choice[trial] = pm.Categorical(pchoice)
#         a = choice[trial]
#
#         RPE = reward[trial] - Q[a]
#
#         Q[a] = Q[a] + alpha * RPE
