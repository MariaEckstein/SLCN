# Helpful example: http://docs.pymc.io/notebooks/multilevel_modeling.html

import numpy as np
import pandas as pd
import glob

import pymc3 as pm
import theano
import theano.tensor as T

import matplotlib.pyplot as plt

# Switches for this script
verbose = True
n_trials = 10
n_subj = 2


# Function definitions
def update_Q(reward, choice, Q_old, alpha):
    return Q_old + choice * alpha * (reward - Q_old)


def p_from_Q(Q_left, Q_right, beta, epsilon):
    p_left = 1 / (1 + T.exp(-beta * (Q_left - Q_right)))  # translate Q-values into probabilities using softmax
    return epsilon * 0.5 + (1 - epsilon) * p_left  # add epsilon noise


# Load to-be-modeled data
filenames = glob.glob('C:/Users/maria/MEGAsync/SLCN/PSGenRecCluster/fit_par/PSsimple_flat*.csv')[:n_subj]
sIDs = range(len(filenames))
right_choices = np.zeros((len(sIDs), n_trials))
rewards = np.zeros(right_choices.shape)

for sID, filename in enumerate(filenames):
    agent_data = pd.read_csv(filename)
    right_choices[sID] = np.array(agent_data['selected_box'])[:n_trials]
    rewards[sID] = agent_data['reward'].tolist()[:n_trials]
left_choices = 1 - right_choices

if verbose:
    print("Left choices:\n", left_choices)
    print("Right choices:\n", right_choices)
    print("Rewards:\n", rewards)

basic_model = pm.Model()
with basic_model:

    # Population-level priors
    alpha_mu = pm.Uniform('alpha_mu', lower=0, upper=1)
    beta_mu = pm.Lognormal('beta_mu', mu=0, sd=1)
    epsilon_mu = pm.Uniform('epsilon_mu', lower=0, upper=1)

    # Individual parameters
    alpha = pm.Normal('alpha', mu=alpha_mu, sd=1)
    beta = pm.Normal('beta', mu=beta_mu, sd=1)
    epsilon = pm.Normal('epsilon', mu=epsilon_mu, sd=1)

    # Observed data
    Q_old = T.as_tensor_variable(0.49999 * np.ones((n_subj, 1)))  # [0.499999])
    Q_old_print = T.printing.Print('Q_old')(Q_old)
    rewards = T.as_tensor_variable(rewards)
    left_choices = T.as_tensor_variable(left_choices)
    right_choices = T.as_tensor_variable(right_choices)

    # Calculate Q-values for both actions
    Q_left, _ = theano.scan(fn=update_Q,
                            sequences=[rewards, left_choices],
                            outputs_info=[Q_old],
                            non_sequences=[alpha])
    Q_right, _ = theano.scan(fn=update_Q,
                             sequences=[rewards, right_choices],
                             outputs_info=[Q_old],
                             non_sequences=[alpha])

    # Get probabilities for the left action
    p_left = p_from_Q(Q_left, Q_right, beta, epsilon)
    shifted_p_left = p_left[:, :-1]
    complete_p_left = T.concatenate([Q_old, shifted_p_left], axis=1)

    # Use Bernoulli to sample responses
    model_choices = pm.Bernoulli('model_choices', p=complete_p_left, observed=left_choices)

    # Print
    if verbose:
        alpha_mu_print = T.printing.Print('alpha_mu')(alpha_mu)
        beta_mu_print = T.printing.Print('beta_mu')(beta_mu)
        epsilon_mu_print = T.printing.Print('epsilon_mu')(epsilon_mu)
        alpha_print = T.printing.Print('alpha')(alpha)
        beta_print = T.printing.Print('beta')(beta)
        epsilon_print = T.printing.Print('epsilon')(epsilon)
        Q_left_print = T.printing.Print('Q_left')(Q_left)
        Q_right_print = T.printing.Print('Q_right')(Q_right)
        p_left_print = T.printing.Print('p_left')(p_left)
        complete_p_left_print = T.printing.Print('complete_p_left')(complete_p_left)
        # model_choices_print = T.printing.Print('model_choices')(model_choices)

    # MCMC samples
    trace = pm.sample(10, tune=1, chains=1, cores=1)

pm.traceplot(trace)
plt.show()
pm.summary(trace).round(2)
