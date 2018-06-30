import glob
import pandas as pd
import numpy as np
import theano
import theano.tensor as T


# Data types of the inputs for update_Q
rewards = T.vector('rewards')
choices = T.vector('choices')
alpha = T.scalar('alpha')
Q_old = 0.5 * T.ones(1)


# update_Q as a single-trial, non-Theano function
# order of arguments: sequences - input from previous iteration - non-sequences
def update_Q_(reward, choice, Q_old, alpha):
    return Q_old + choice * alpha * (reward - Q_old)


# Get scan output and updates for update_Q, telling it how the loop will be performed
output, updates = theano.scan(fn=update_Q_,  # function that will executed in each iteration, with the following inputs:
                              sequences=[rewards, choices],  # parameter with a different value in each iteration
                              outputs_info=[Q_old],  # parameter that will be updated in each iteration
                              non_sequences=[alpha])  # parameter with the same value across iterations

# Make update_Q a scan function, encompassing the loop
update_Q = theano.function(inputs=[rewards, choices, alpha],
                           outputs=output,
                           updates=updates)

# # Test on agent data (choices, rewards)
# n_trials = 5
# agent_data = pd.read_csv(glob.glob('C:/Users/maria/MEGAsync/SLCN/PSGenRec/*.csv')[0])
# left_choices = np.array(agent_data['selected_box'])[:n_trials]
# right_choices = 1 - np.array(agent_data['selected_box'])[:n_trials]
# rewards = agent_data['reward'].tolist()[:n_trials]
# alpha = 0.5
#
# print(choices)
# print(rewards)
# print(update_Q(rewards, right_choices, alpha))