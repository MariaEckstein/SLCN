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
fitting_method = 'hierarchical'  # 'hierarchical', 'flat'
fit_RL = True
fit_Bayes = True
compare_RL_Bayes = True

# Which data should be fitted?
fitted_data_name = 'humans'  # 'humans', 'simulations'

# Sampling details
n_samples = 10000
n_tune = 200
n_chains = 2

# Get data path and save path
shared = Shared()
if fitted_data_name == 'humans':
    learning_style = ''
    data_dir = shared.get_paths()['human data']
    file_name_pattern = 'PS*.csv'
    n_trials = 128
    n_subj = 500
else:
    learning_style = 'hierarchical'
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
ages = np.zeros(n_subj)

# Load data and bring in the right format
SLCNinfo = pd.read_csv(shared.get_paths()['ages file name'])
for file_idx, filename in enumerate(filenames):
    agent_data = pd.read_csv(filename)
    if agent_data.shape[0] > n_trials:
        choices[:, file_idx] = np.array(agent_data['selected_box'])[:n_trials]
        rewards[:, file_idx] = agent_data['reward'].tolist()[:n_trials]
        sID = agent_data['sID'][0]
        ages[file_idx] = SLCNinfo[SLCNinfo['ID'] == sID]['PreciseYrs'].values

# Remove excess columns
rewards = np.delete(rewards, range(file_idx+1, n_subj), 1)
choices = np.delete(choices, range(file_idx+1, n_subj), 1)
ages = ages[:file_idx+1]
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

with pm.Model() as Bayes_model:

    # Observed data (choices & rewards)
    rewards = T.as_tensor_variable(rewards)
    choices = T.as_tensor_variable(choices)
    ages = T.as_tensor_variable(ages)
    initial_p = 0.5 * T.ones((1, n_subj))  # get first entry

    # Model parameters
    if fitting_method == 'hierarchical':

        # Population-level priors (as un-informative as possible)
        epsilon_mu, epsilon_sd = pm.Uniform('epsilon_mu', lower=0, upper=1), 0.2
        p_switch_mu, p_switch_sd = pm.Uniform('p_switch_mu', lower=0, upper=1), 0.1  # pm.HalfNormal('p_switch_sd', sd=0.1)
        p_switch_slope = T.as_tensor_variable(pm.Uniform('p_switch_slope', lower=-1, upper=1))
        p_reward_mu, p_reward_sd = pm.Uniform('p_reward_mu', lower=0, upper=1), 0.1  # pm.HalfNormal('p_reward_sd', sd=0.1)
        p_reward_slope = T.as_tensor_variable(pm.Uniform('p_reward_slope', lower=-1, upper=1))
        # p_noisy_mu, p_noisy_sd = pm.Uniform('p_noisy_mu', lower=0, upper=1), 0.1  # pm.HalfNormal('p_noisy_sd', sd=0.1)

        # Individual parameters (bounds avoid initial energy==-inf because logp(RV)==-inf)
        epsilon = pm.Bound(pm.Normal, lower=0, upper=1)('epsilon', mu=epsilon_mu, sd=epsilon_sd, shape=n_subj)
        p_switch = pm.Bound(pm.Normal, lower=0, upper=1)('p_switch',
                                                         mu=p_switch_mu + ages * p_switch_slope, sd=p_switch_sd, shape=n_subj)  #
        T.printing.Print(p_switch)
        p_reward = pm.Bound(pm.Normal, lower=0, upper=1)('p_reward',
                                                         mu=p_reward_mu + ages * p_reward_slope, sd=p_reward_sd, shape=n_subj)  #
        T.printing.Print(p_reward)
        p_noisy = 0.09 * T.ones(n_subj)  # pm.Bound(pm.Normal, lower=0, upper=1)('p_noisy', mu=p_noisy_mu, sd=p_noisy_sd, shape=n_subj)

    elif fitting_method == 'flat':

        # Individual parameters
        epsilon = pm.Uniform('epsilon', lower=0, upper=1, shape=n_subj)
        p_switch = pm.Uniform('p_switch', lower=0, upper=1, shape=n_subj)
        p_reward = pm.Uniform('p_reward', lower=0, upper=1, shape=n_subj)
        p_noisy = pm.Uniform('p_noisy', lower=0, upper=1, shape=n_subj)

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
        print("Test point: {0}".format(Bayes_model.test_point))
        print("\tmodel.logp(Bayes_model.test_point): {0}".format(Bayes_model.logp(Bayes_model.test_point)))
        for RV in Bayes_model.basic_RVs:
            print("\tlogp of {0}: {1}".format(RV.name, RV.logp(Bayes_model.test_point)))

    # Sample the model (should be >=5000, tune>=500)
    Bayes_trace = pm.sample(n_samples, tune=n_tune, chains=n_chains, cores=1)

# Show results
pm.traceplot(Bayes_trace)
Bayes_model_summary = pm.summary(Bayes_trace)

# Save results
now = datetime.datetime.now()
Bayes_model_id = '_'.join([str(now.year), str(now.month), str(now.day), str(now.hour), str(now.minute),
                     fitted_data_name, "Bayes", fitting_method, 'n_samples' + str(n_samples)])
print('Pickling Bayes_trace and Bayes_model, saving traceplot to {0}{1}...\n'.format(save_dir, Bayes_model_id))
with open(save_dir + Bayes_model_id + '.pickle', 'wb') as handle:
    pickle.dump({'trace': Bayes_trace, 'model': Bayes_model}, handle, protocol=pickle.HIGHEST_PROTOCOL)
plt.savefig(save_dir + 'plot_' + Bayes_model_id + '.png')

with pm.Model() as RL_model:

    # Observed data (choices & rewards)
    rewards = T.as_tensor_variable(rewards)
    choices = T.as_tensor_variable(choices)
    ages = T.as_tensor_variable(ages)
    initial_p = 0.5 * T.ones((1, n_subj))  # get first entry

    # Model parameters
    if fitting_method == 'hierarchical':

        # Population-level priors (as un-informative as possible)
        epsilon_mu, epsilon_sd = pm.Uniform('epsilon_mu', lower=0, upper=1), 0.2
        epsilon_slope = pm.Uniform('epsilon_slope', lower=-1, upper=1)
        alpha_mu, alpha_sd = pm.Uniform('alpha_mu', lower=0, upper=1), 0.1  # pm.HalfNormal('alpha_sd', sd=0.1)
        alpha_slope = pm.Uniform('alpha_slope', lower=-1, upper=1)
        calpha_sc_mu, calpha_sc_sd = pm.Uniform('calpha_sc_mu', lower=0, upper=1), 0.1  # pm.HalfNormal('calpha_sc_sd', sd=0.1)
        calpha_slope = pm.Uniform('calpha_slope', lower=-1, upper=1)
        beta_mu, beta_sd = pm.Lognormal('beta_mu', mu=0, sd=1), 3  # pm.HalfNormal('beta_sd', sd=0.1)
        beta_slope = pm.Uniform('beta_slope', lower=-1, upper=1)

        # Individual parameters (bounds avoid initial energy==-inf because logp(RV)==-inf)
        epsilon = pm.Bound(pm.Normal, lower=0, upper=1)('epsilon', mu=epsilon_mu + ages * epsilon_slope, sd=epsilon_sd, shape=n_subj)
        alpha = pm.Bound(pm.Normal, lower=0, upper=1)('alpha', mu=alpha_mu + ages * alpha_slope, sd=alpha_sd, shape=n_subj)
        calpha_sc = pm.Bound(pm.Normal, lower=0, upper=1)('calpha_sc', mu=calpha_sc_mu + ages * calpha_slope, sd=alpha_sd, shape=n_subj)
        calpha = pm.Deterministic('calpha', alpha * calpha_sc)
        beta = pm.Normal('beta', mu=beta_mu + beta_slope, sd=beta_sd, shape=n_subj)

    elif fitting_method == 'flat':

        # Individual parameters
        epsilon = pm.Uniform('epsilon', lower=0, upper=1, shape=n_subj)
        alpha = pm.Uniform('alpha', lower=0, upper=1, shape=n_subj)
        calpha_sc = pm.Uniform('calpha_sc', lower=0, upper=1, shape=n_subj)
        calpha = alpha * calpha_sc
        beta = pm.Lognormal('beta', mu=0, sd=1, shape=n_subj)

    # Calculate Q-values
    initial_Q_left = 0.5 * T.ones(n_subj)
    initial_Q_right = 0.5 * T.ones(n_subj)
    [Q_left, Q_right], _ = theano.scan(fn=shared.update_Q,
                                       sequences=[rewards, choices],
                                       outputs_info=[initial_Q_left, initial_Q_right],
                                       non_sequences=[alpha, calpha])

    # Translate Q-values into probabilities and add epsilon noise
    p_right = shared.p_from_Q(Q_left, Q_right, beta, epsilon)

    p_right = T.concatenate([initial_p, p_right[:-1]], axis=0)  # add initial p=0.5 at the beginning

    # Use Bernoulli to sample responses
    model_choices = pm.Bernoulli('model_choices', p=p_right, observed=choices)
    # action_prob = p_right * choices + (1 - p_right) * (1 - choices)
    # LL += np.log(action_prob)

    # Check model logp and RV logps (will crash if they are nan or -inf)
    if verbose or print_logps:
        print("Checking that none of the logp are -inf:")
        print("Test point: {0}".format(RL_model.test_point))
        print("\tRL_model.logp(RL_model.test_point): {0}".format(RL_model.logp(RL_model.test_point)))
        for RV in RL_model.basic_RVs:
            print("\tlogp of {0}: {1}".format(RV.name, RV.logp(RL_model.test_point)))

    # Sample the model (should be >=5000, tune>=500)
    RL_trace = pm.sample(n_samples, tune=n_tune, chains=n_chains, cores=1)

# Show results
pm.traceplot(RL_trace)
RL_model_summary = pm.summary(RL_trace)

# Save results
now = datetime.datetime.now()
RL_model_id = '_'.join([str(now.year), str(now.month), str(now.day), str(now.hour), str(now.minute),
                     fitted_data_name, "RL", fitting_method, 'n_samples' + str(n_samples)])
print('Pickling RL_trace and RL_model, saving traceplot to {0}{1}...\n'.format(save_dir, RL_model_id))
with open(save_dir + RL_model_id + '.pickle', 'wb') as handle:
    pickle.dump({'trace': RL_trace, 'model': RL_model}, handle, protocol=pickle.HIGHEST_PROTOCOL)
plt.savefig(save_dir + 'plot_' + RL_model_id + '.png')

# Compare WAIC scores
print("Comparing models...")
RL_model.name = 'RL_model'
Bayes_model.name = 'Bayes_model'
df_comp_WAIC = pm.compare({RL_model: RL_trace, Bayes_model: Bayes_trace})

pm.compareplot(df_comp_WAIC)
plt.savefig(save_dir + 'compareplot_WAIC.png')

# Compare leave-one-out cross validation
df_comp_LOO = pm.compare({RL_model: RL_trace, Bayes_model: Bayes_trace}, ic='LOO')

pm.compareplot(df_comp_LOO)
plt.savefig(save_dir + 'compareplot_LOO.png')

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
