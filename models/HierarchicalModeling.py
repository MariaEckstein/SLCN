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
n_trials = 200
n_subj = 10
model_name = 'Bayes'
fit_hierarchical = False

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
    print("Loaded {0} datasets with pattern {1} from {2}.".format(n_subj, file_name_pattern, data_dir))
    print("Right choices - shape: {0}\n{1}\n".format(choices.shape, choices))
    print("Rewards - shape: {0}\n{1}\n".format(rewards.shape, rewards))

# Fit model
print("Compiling the model...")
with pm.Model() as model:

    if fit_hierarchical:

        # Population-level priors (as un-informative as possible)
        alpha_mu = pm.Uniform('alpha_mu', lower=0, upper=1)
        calpha_scaler_mu = pm.Uniform('calpha_scaler_mu', lower=0, upper=1)
        beta_mu = pm.Lognormal('beta_mu', mu=0, sd=1)
        epsilon_mu = pm.Uniform('epsilon_mu', lower=0, upper=1)

        alpha_sd = 0.1  # pm.HalfNormal('alpha_sd', sd=0.5)
        beta_sd = 3  # pm.HalfNormal('beta_sd', sd=3)
        epsilon_sd = 0.1  # pm.HalfNormal('epsilon_sd', sd=0.5)

        # Individual parameters (specify bounds to avoid initial energy==-inf because logp(RV)==-inf)
        alpha = pm.Bound(pm.Normal, lower=0, upper=1)('alpha', mu=alpha_mu, sd=alpha_sd, shape=n_subj)
        calpha_scaler = pm.Bound(pm.Normal, lower=0, upper=1)('calpha_scaler',
                                                              mu=calpha_scaler_mu, sd=alpha_sd, shape=n_subj)
        calpha = pm.Deterministic('calpha', alpha * calpha_scaler)
        beta = pm.Bound(pm.Normal, lower=0)('beta', mu=beta_mu, sd=beta_sd, shape=n_subj)
        epsilon = pm.Bound(pm.Normal, lower=0, upper=1)('epsilon', mu=epsilon_mu, sd=epsilon_sd, shape=n_subj)

    else:

        # Individual parameters
        epsilon = pm.Uniform('epsilon', lower=0, upper=1, shape=n_subj)
        if model_name == 'RL':
            alpha = pm.Uniform('alpha', lower=0, upper=1, shape=n_subj)
            calpha_scaler = pm.Uniform('calpha_scaler', lower=0, upper=1, shape=n_subj)
            calpha = pm.Deterministic('calpha', alpha * calpha_scaler)
            beta = pm.Lognormal('beta', mu=0, sd=1, shape=n_subj)

        elif model_name == 'Bayes':
            p_switch = pm.Uniform('p_switch', lower=0, upper=1, shape=n_subj)  # pm.Bound(pm.Normal, lower=0, upper=1)('p_switch', mu=0.1, sd=0.01, shape=n_subj)  # 0.1 * T.ones(n_subj)  #
            # p_reward = pm.Uniform('p_reward', lower=0, upper=1, shape=n_subj)  # pm.Bound(pm.Normal, lower=0, upper=1)('p_reward', mu=0.75, sd=0.1, shape=n_subj)
            p_reward = 0.75 * T.ones(n_subj)
            # p_noisy_task = pm.Uniform('p_noisy_task', lower=0, upper=1, shape=n_subj)  # pm.Bound(pm.Normal, lower=0, upper=1)('p_noisy_task', mu=0.01, sd=0.01, shape=n_subj)  # 0.01 * T.ones(n_subj)  #
            p_noisy_task = 0.01 * T.ones(n_subj)

    # Observed data (choices & rewards)
    rewards = T.as_tensor_variable(rewards)
    choices = T.as_tensor_variable(choices)
    initial_p = T.as_tensor_variable(0.5 * np.ones((1, n_subj)))  # get first entry

    if model_name == 'RL':

        # Calculate Q-values
        initial_Q_left = T.as_tensor_variable(0.5 * np.ones(n_subj))
        initial_Q_right = T.as_tensor_variable(0.5 * np.ones(n_subj))
        [Q_left, Q_right], _ = theano.scan(fn=shared.update_Q,
                                           sequences=[rewards, choices],
                                           outputs_info=[initial_Q_left, initial_Q_right],
                                           non_sequences=[alpha, calpha])

        # Translate Q-values into probabilities
        p_right = shared.p_from_Q(Q_left, Q_right, beta)
        p_right = shared.add_epsilon_noise(p_right, epsilon)[:-1]  # remove last entry

    elif model_name == 'Bayes':

        if verbose:
            T.printing.Print('choices')(choices)
            T.printing.Print('rewards')(rewards)

        # Get likelihoods
        lik_cor, lik_inc = shared.get_likelihoods(rewards, choices, p_reward, p_noisy_task)

        # Get posterior
        p_right = T.as_tensor_variable(0.5 * np.ones(n_subj))
        p_right, _ = theano.scan(fn=shared.post_from_lik,
                                 sequences=[lik_cor, lik_inc],
                                 outputs_info=[p_right])

        # Get probability for subsequent trial
        p_right = shared.get_p_subsequent_trial(p_right, p_switch)

        # Add epsilon noise
        p_right = shared.add_epsilon_noise(p_right, epsilon)[:-1]  # remove last entry

    p_right = T.concatenate([initial_p, p_right], axis=0)  # combine

    # Use Bernoulli to sample responses
    model_choices = pm.Bernoulli('model_choices', p=p_right, observed=choices)

    # Check model logp and RV logps (will crash if they are nan or -inf)
    if verbose:
        print("model.logp(model.test_point): {0}".format(model.logp(model.test_point)))
        for RV in model.basic_RVs:
            print("logp of {0}: {1}".format(RV.name, RV.logp(model.test_point)))

    # Sample the model (should be >=5000, tune>=500)
    trace = pm.sample(5000, tune=50, chains=1, cores=1)

# Show results
pm.traceplot(trace)
model_summary = pm.summary(trace)

# Save results
now = datetime.datetime.now()
model_id = str(now.year) + '_' + str(now.month) + '_' + str(now.day) + '_nsubj' + str(n_subj) + 'Bayes4'
print('Pickling trace, model, and plot to {0}{1}'.format(save_dir, model_id))
with open(save_dir + 'trace_' + model_id + '.pickle', 'wb') as handle:
    pickle.dump(trace, handle)
with open(save_dir + 'model_' + model_id + '.pickle', 'wb') as handle:
    pickle.dump(model, handle)
plt.savefig(save_dir + 'plot_' + model_id + '.png')

# NOTES
# Hierarchical fitting example: http://docs.pymc.io/notebooks/multilevel_modeling.html
# And: http://docs.pymc.io/notebooks/rugby_analytics.html
# with pm.Model() as model:
#     # global model parameters
#     home = pm.Flat('home')
#     sd_att = pm.HalfStudentT('sd_att', nu=3, sd=2.5)
#     sd_def = pm.HalfStudentT('sd_def', nu=3, sd=2.5)
#     intercept = pm.Flat('intercept')
#
#     # team-specific model parameters
#     atts_star = pm.Normal("atts_star", mu=0, sd=sd_att, shape=num_teams)
#     defs_star = pm.Normal("defs_star", mu=0, sd=sd_def, shape=num_teams)
#
#     atts = pm.Deterministic('atts', atts_star - tt.mean(atts_star))
#     defs = pm.Deterministic('defs', defs_star - tt.mean(defs_star))
#     home_theta = tt.exp(intercept + home + atts[home_team] + defs[away_team])
#     away_theta = tt.exp(intercept + atts[away_team] + defs[home_team])
#
#     # likelihood of observed data
#     home_points = pm.Poisson('home_points', mu=home_theta, observed=observed_home_goals)
#     away_points = pm.Poisson('away_points', mu=away_theta, observed=observed_away_goals)

# Problem with "bad initial energy error":
# logp(RV) == -inf and/or logp(model) == -inf because Bernoulli can't handle negative probabilities, which can arise
# in Q_from_P when parameters are unbounded; see https://discourse.pymc.io/t/frequently-asked-questions/74)

# # Mimic loop instantiated by theano.scan for debugging
# Q_old = 0.499 * np.ones((n_subj))
# alpha = 0.1 * np.ones((n_subj))
# for reward, choice in zip(rewards, left_choices):
#     print("Q_old:", Q_old)
#     print("Choice:", choice)
#     print("Reward:", reward)
#     Q_old = shared.update_Q(reward, choice, Q_old, alpha)

# Starting to think about our hierarchical models:
# https://discourse.pymc.io/t/filtering-e-g-particle-filter-sequential-mc/117
# https://gist.github.com/fonnesbeck/342989 (old - prob not useful)
# https://github.com/hstrey/Hidden-Markov-Models-pymc3/blob/master/Multi-State%20HMM.ipynb (looks better!)
# https://stackoverflow.com/questions/42146225/problems-building-a-discrete-hmm-in-pymc3
# https://stackoverflow.com/questions/19875621/hidden-markov-in-pymc3/20288474
