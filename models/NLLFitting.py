import numpy as np
import pandas as pd

import glob
import pickle
import datetime
import os

import pymc3 as pm
import theano
import theano.tensor as T
from shared_modeling_simulation import Shared

import matplotlib.pyplot as plt


# Switches for this script
verbose = False
print_logps = True
n_trials = 20
n_subj = 3
model_name = 'Bayes'
fitting_method = 'flat'  # 'hiearchical', 'flat'

# Get data path and save path
shared = Shared()
# data_dir = shared.get_paths()['human data']
data_dir = shared.get_paths()['simulations']
file_name_pattern = 'PS' + model_name + '*.csv'
save_dir = shared.get_paths()['fitting results']
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Prepare things for loading data
filenames = glob.glob(data_dir + file_name_pattern)[:n_subj]
assert len(filenames) > 0, "Error: There are no files with pattern {0} in {1}".format(file_name_pattern, data_dir)
choices = np.zeros((n_trials, len(filenames)))
rewards = np.zeros(choices.shape)

# Load data and bring in the right format
for sID, filename in enumerate(filenames):
    agent_data = pd.read_csv(filename)
    if agent_data.shape[0] > n_trials:
        choices[:, sID] = np.array(agent_data['selected_box'])[:n_trials]
        rewards[:, sID] = agent_data['reward'].tolist()[:n_trials]

# Remove excess columns
rewards = np.delete(rewards, range(sID+1, n_subj), 1)
choices = np.delete(choices, range(sID+1, n_subj), 1)
n_subj = choices.shape[1]

# Look at data
if verbose:
    print("Loaded {0} datasets with pattern {1} from {2}\n.".format(n_subj, file_name_pattern, data_dir))
    print("Choices - shape: {0}\n{1}\n".format(choices.shape, choices))
    print("Rewards - shape: {0}\n{1}\n".format(rewards.shape, rewards))

# Fit model
eps = np.arange(0, 1, 0.05)
LLs = np.zeros((len(eps), n_subj))

LL = np.zeros(n_subj)
for i, epsi in enumerate(eps):

    #DEBUG
    epsi = 0.1

    # Individual parameters
    epsilon = epsi * np.ones(n_subj)
    # epsilon = pm.Uniform('epsilon', lower=0, upper=1, shape=n_subj)
    if model_name == 'RL':
        alpha = pm.Uniform('alpha', lower=0, upper=1, shape=n_subj)
        calpha_scaler = pm.Uniform('calpha_scaler', lower=0, upper=1, shape=n_subj)
        calpha = pm.Deterministic('calpha', alpha * calpha_scaler)
        beta = pm.Lognormal('beta', mu=0, sd=1, shape=n_subj)

    elif model_name == 'Bayes':
        # p_switch = pm.Uniform('p_switch', lower=0, upper=1, shape=n_subj)  # pm.Bound(pm.Normal, lower=0, upper=1)('p_switch', mu=0.1, sd=0.01, shape=n_subj)  # 0.1 * T.ones(n_subj)  #
        p_switch = 0.1 * np.ones(n_subj)
        # p_reward = pm.Uniform('p_reward', lower=0.1, upper=0.9, shape=n_subj)  # pm.Bound(pm.Normal, lower=0, upper=1)('p_reward', mu=0.75, sd=0.1, shape=n_subj)
        p_reward = 0.75 * np.ones(n_subj)
        # p_noisy_task = pm.Uniform('p_noisy_task', lower=0, upper=1, shape=n_subj)  # pm.Bound(pm.Normal, lower=0, upper=1)('p_noisy_task', mu=0.01, sd=0.01, shape=n_subj)  # 0.01 * np.ones(n_subj)  #
        p_noisy_task = 0.01 * np.ones(n_subj)

    initial_p = 0.5 * np.ones((1, n_subj))  # get first entry

    if model_name == 'RL':

        # Calculate Q-values
        initial_Q_left = 0.5 * np.ones(n_subj)
        initial_Q_right = 0.5 * np.ones(n_subj)
        [Q_left, Q_right], _ = theano.scan(fn=shared.update_Q,
                                           sequences=[rewards, choices],
                                           outputs_info=[initial_Q_left, initial_Q_right],
                                           non_sequences=[alpha, calpha])

        # Translate Q-values into probabilities
        p_right = shared.p_from_Q(Q_left, Q_right, beta)
        p_right = shared.add_epsilon_noise(p_right, epsilon)

    elif model_name == 'Bayes':

        # Get likelihoods
        lik_cor, lik_inc = shared.get_likelihoods(rewards, choices, p_reward, p_noisy_task)

        # Get posterior
        p_right = 0.5 * np.ones((rewards.shape[0], n_subj))
        pr = 0.5 * np.ones(n_subj)
        for trial, (lik_cor_trial, lik_inc_trial) in enumerate(zip(lik_cor, lik_inc)):
            pr = shared.post_from_lik(lik_cor_trial, lik_inc_trial, pr)

            # Get probability for subsequent trial
            pr = shared.get_p_subsequent_trial(pr, p_switch)

            # Add epsilon noise
            pr = shared.add_epsilon_noise(pr, epsilon)

            p_right[trial, :] = pr
    p_right = np.concatenate([initial_p, p_right[:-1]], axis=0)  # add initial p=0.5 at the beginning

    # Use Bernoulli to sample responses
    action_prob = p_right * choices + (1 - p_right) * (1 - choices)
    LL = np.log(action_prob)
    LLs = np.cumsum(LL, axis=0)

    print(LLs)
