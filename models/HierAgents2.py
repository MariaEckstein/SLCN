import pymc3 as pm
import glob
import pandas as pd
import numpy as np
import theano
import theano.tensor as T

import seaborn as sns
import matplotlib.pyplot as plt


# Define update_Q and p_from_Q as Theano scan functions to get around the loop over trials
from update_Q import update_Q, p_from_Q
n_trials = 5

# Get to-be-modeled data (agent's choices & obtained rewards)
agent_data = pd.read_csv(glob.glob('C:/Users/maria/MEGAsync/SLCN/PSGenRec/*.csv')[0])
left_choices = 1 - np.array(agent_data['selected_box'])[:n_trials]
right_choices = np.array(agent_data['selected_box'])[:n_trials]
rewards = agent_data['reward'].tolist()[:n_trials]

with pm.Model() as model:

    # Specify model parameters
    epsilon = pm.Uniform('epsilon', lower=0, upper=1)
    alpha = pm.Uniform('alpha', lower=0, upper=1)
    beta = 2  # pm.Uniform('beta', lower=0, upper=5)

    # Calculate Q-values of left button and right button
    Q_left = update_Q(rewards, left_choices, alpha).T[0]
    # Q_right = update_Q(rewards, right_choices, alpha).T[0]
    #
    # # Transform Q-values into action probabilities using a softmax transform
    # p_left = p_from_Q(Q_left, Q_right, beta)
    # p_right = p_from_Q(Q_right, Q_left, beta)
    #
    # # print(np.column_stack((Q_left, Q_right, p_left, p_right, p_left + p_right, rewards, left_choices)))
    # print(np.column_stack((p_left, p_right)))
    #
    # # Observed choice
    # observed_choice = pm.Bernoulli('observed_choice', p=np.column_stack((p_left, p_right)), observed=np.column_stack((right_choices, left_choices)))

    observed_choice = pm.Bernoulli('observed_choice', p=epsilon, observed=right_choices)

    # Inference
    trace = pm.sample(1000, tune=500, cores=1)

# Plot results
pm.traceplot(trace)
plt.show()
