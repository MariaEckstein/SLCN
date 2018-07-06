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
print_logps = False

# Which model should be run?
learning_style = 'RL'  # 'RL' or 'Bayes'
fitting_method = 'hierarchical'  # 'hierarchical', 'flat'

# Which data should be fitted?
fitted_data_name = 'humans'  # 'humans', 'simulations'

# Sampling details
n_samples = 10000
n_tune = 200
n_chains = 2

# Get data path and save path
shared = Shared()
if fitted_data_name == 'humans':
    data_dir = shared.get_paths()['human data']
    file_name_pattern = 'PS*.csv'
    n_trials = 128
    n_subj = 500
else:
    data_dir = shared.get_paths()['simulations']
    file_name_pattern = 'PS' + learning_style + '*.csv'
    n_trials = 200
    n_subj = 50
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
print("Loaded {0} datasets with pattern {1} from {2}...\n".format(n_subj, file_name_pattern, data_dir))
if verbose:
    print("Choices - shape: {0}\n{1}\n".format(choices.shape, choices))
    print("Rewards - shape: {0}\n{1}\n".format(rewards.shape, rewards))

# Fit model
# LL = np.zeros(n_subj)
print("Compiling {0} {1} model for {2} with {3} samples and {4} tuning steps...\n".format(
    fitting_method, learning_style, fitted_data_name, n_samples, n_tune))

with pm.Model() as model:

    if fitting_method == 'hierarchical':

        # Population-level priors (as un-informative as possible)
        epsilon_mu, epsilon_sd = pm.Uniform('epsilon_mu', lower=0, upper=1), 0.2

        if learning_style == 'RL':
            alpha_mu, alpha_sd = pm.Uniform('alpha_mu', lower=0, upper=1), 0.1  # pm.HalfNormal('alpha_sd', sd=0.1)
            calpha_sc_mu, calpha_sc_sd = pm.Uniform('calpha_sc_mu', lower=0, upper=1), 0.1  # pm.HalfNormal('calpha_sc_sd', sd=0.1)
            beta_mu, beta_sd = pm.Lognormal('beta_mu', mu=0, sd=1), 3  # pm.HalfNormal('beta_sd', sd=0.1)

        elif learning_style == 'Bayes':
            p_switch_mu, p_switch_sd = pm.Uniform('p_switch_mu', lower=0, upper=1), 0.1  # pm.HalfNormal('p_switch_sd', sd=0.1)
            p_reward_mu, p_reward_sd = pm.Uniform('p_reward_mu', lower=0, upper=1), 0.1  # pm.HalfNormal('p_reward_sd', sd=0.1)
            p_noisy_mu, p_noisy_sd = pm.Uniform('p_noisy_mu', lower=0, upper=1), 0.1  # pm.HalfNormal('p_noisy_sd', sd=0.1)

        # Individual parameters (bounds avoid initial energy==-inf because logp(RV)==-inf)
        epsilon = pm.Bound(pm.Normal, lower=0, upper=1)('epsilon', mu=epsilon_mu, sd=epsilon_sd, shape=n_subj)

        if learning_style == 'RL':
            alpha = pm.Bound(pm.Normal, lower=0, upper=1)('alpha', mu=alpha_mu, sd=alpha_sd, shape=n_subj)
            calpha_sc = pm.Bound(pm.Normal, lower=0, upper=1)('calpha_sc', mu=calpha_sc_mu, sd=alpha_sd, shape=n_subj)
            calpha = pm.Deterministic('calpha', alpha * calpha_sc)
            beta = pm.Normal('beta', mu=beta_mu, sd=beta_sd, shape=n_subj)

        elif learning_style == 'Bayes':
            p_switch = pm.Bound(pm.Normal, lower=0, upper=1)('p_switch', mu=p_switch_mu, sd=p_switch_sd, shape=n_subj)
            p_reward = pm.Bound(pm.Normal, lower=0, upper=1)('p_reward', mu=p_reward_mu, sd=p_reward_sd, shape=n_subj)
            p_noisy = pm.Bound(pm.Normal, lower=0, upper=1)('p_noisy', mu=p_noisy_mu, sd=p_noisy_sd, shape=n_subj)

    elif fitting_method == 'flat':

        # Individual parameters
        epsilon = pm.Uniform('epsilon', lower=0, upper=1, shape=n_subj)
        if learning_style == 'RL':
            alpha = pm.Uniform('alpha', lower=0, upper=1, shape=n_subj)
            calpha_sc = pm.Uniform('calpha_sc', lower=0, upper=1, shape=n_subj)
            calpha = alpha * calpha_sc
            beta = pm.Lognormal('beta', mu=0, sd=1, shape=n_subj)

        elif learning_style == 'Bayes':
            p_switch = pm.Uniform('p_switch', lower=0, upper=1, shape=n_subj)
            p_reward = pm.Uniform('p_reward', lower=0, upper=1, shape=n_subj)
            p_noisy = pm.Uniform('p_noisy', lower=0, upper=1, shape=n_subj)

    # Observed data (choices & rewards)
    rewards = T.as_tensor_variable(rewards)
    choices = T.as_tensor_variable(choices)
    initial_p = 0.5 * T.ones((1, n_subj))  # get first entry

    if learning_style == 'RL':

        # Calculate Q-values
        initial_Q_left = 0.5 * T.ones(n_subj)
        initial_Q_right = 0.5 * T.ones(n_subj)
        [Q_left, Q_right], _ = theano.scan(fn=shared.update_Q,
                                           sequences=[rewards, choices],
                                           outputs_info=[initial_Q_left, initial_Q_right],
                                           non_sequences=[alpha, calpha])

        # Translate Q-values into probabilities and add epsilon noise
        p_right = shared.p_from_Q(Q_left, Q_right, beta, epsilon)

    elif learning_style == 'Bayes':

        # Get likelihoods
        lik_cor, lik_inc = shared.get_likelihoods(rewards, choices, p_reward, p_noisy)

        # Get posterior, calculate probability of subsequent trial, add epsilon noise
        p_right = 0.5 * T.ones(n_subj)
        p_right, _ = theano.scan(fn=shared.post_from_lik,
                                 sequences=[lik_cor, lik_inc],
                                 outputs_info=[p_right],
                                 non_sequences=[p_switch, epsilon])

    p_right = T.concatenate([initial_p, p_right[:-1]], axis=0)  # add initial p=0.5 at the beginning

    # Use Bernoulli to sample responses
    model_choices = pm.Bernoulli('model_choices', p=p_right, observed=choices)
    # action_prob = p_right * choices + (1 - p_right) * (1 - choices)
    # LL += np.log(action_prob)

    # Check model logp and RV logps (will crash if they are nan or -inf)
    if verbose or print_logps:
        print("Checking that none of the logp are -inf:")
        print("Test point: {0}".format(model.test_point))
        print("\tmodel.logp(model.test_point): {0}".format(model.logp(model.test_point)))
        for RV in model.basic_RVs:
            print("\tlogp of {0}: {1}".format(RV.name, RV.logp(model.test_point)))

    # Sample the model (should be >=5000, tune>=500)
    trace = pm.sample(n_samples, tune=n_tune, chains=n_chains, cores=1)

# Show results
pm.traceplot(trace)
model_summary = pm.summary(trace)

# Save results
now = datetime.datetime.now()
model_id = '_'.join([str(now.year), str(now.month), str(now.day), str(now.hour), str(now.minute),
                     fitted_data_name, learning_style, fitting_method, 'n_samples' + str(n_samples)])
print('Pickling trace and model, saving traceplot to {0}{1}...\n'.format(save_dir, model_id))
with open(save_dir + model_id + '.pickle', 'wb') as handle:
    pickle.dump({'trace': trace, 'model': model}, handle, protocol=pickle.HIGHEST_PROTOCOL)
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
