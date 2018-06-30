import pymc3 as pm
import glob
import pandas as pd
import numpy as np
import theano
import theano.tensor as T

import seaborn as sns
import matplotlib.pyplot as plt


# Step 1: Define update_Q in Theano scan to avoid the loop over trials
from update_Q import *

# Get agent data (choices, rewards)
n_trials = 5
agent_data = pd.read_csv(glob.glob('C:/Users/maria/MEGAsync/SLCN/PSGenRec/*.csv')[0])
left_choices = np.array(agent_data['selected_box'])[:n_trials]
right_choices = 1 - np.array(agent_data['selected_box'])[:n_trials]
rewards = agent_data['reward'].tolist()[:n_trials]
alpha = 0.5

print(choices)
print(rewards)
print(update_Q(rewards, right_choices, alpha))


Q = T.vector('Q')
beta = T.scalar('beta')

def p_from_Q_(Q, beta):
    return 1 / (1 + T.exp(-beta * (Q[1] - Q[0])))

output, updates = theano.scan(fn=p_from_Q_,
                              sequences=[Q],
                              non_sequences=[beta])
#
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
