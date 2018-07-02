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

import numpy as np
import pandas as pd

import glob
import pickle
import datetime

import pymc3 as pm
import theano
import theano.tensor as T

import matplotlib.pyplot as plt


# Switches for this script
verbose = False
n_trials = 129
base_dir = 'C:/Users/maria/MEGAsync/SLCN/PShumanData/'
data_dir = base_dir + '/PS_*.csv'
n_subj = len(glob.glob(data_dir))


# Define RL model functions
def update_Q(reward, choice, Q_old, alpha):
    return Q_old + choice * alpha * (reward - Q_old)


def p_from_Q(Q_left, Q_right, beta, epsilon):
    p_left = 1 / (1 + np.exp(-beta * (Q_left - Q_right)))  # translate Q-values into probabilities using softmax
    return epsilon * 0.5 + (1 - epsilon) * p_left  # add epsilon noise


# Load to-be-modeled data
filenames = glob.glob(data_dir)[:n_subj]
right_choices = np.zeros((n_trials, len(filenames)))
rewards = np.zeros(right_choices.shape)

for sID, filename in enumerate(filenames):
    agent_data = pd.read_csv(filename)
    if agent_data.shape[0] > n_trials:
        right_choices[:, sID] = np.array(agent_data['selected_box'])[:n_trials]
        rewards[:, sID] = agent_data['reward'].tolist()[:n_trials]
rewards = np.delete(rewards, range(sID, n_subj), 1)
right_choices = np.delete(right_choices, range(sID, n_subj), 1)
left_choices = 1 - right_choices
n_subj = right_choices.shape[1]

# Look at data
if verbose:
    print("Number of datasets:", n_subj)
    print("Left choices - shape: {0}\n{1}\n".format(left_choices.shape, left_choices))
    print("Right choices - shape: {0}\n{1}\n".format(right_choices.shape, right_choices))
    print("Rewards - shape: {0}\n{1}\n".format(rewards.shape, rewards))

# # Mimic loop instantiated by theano.scan for debugging
# Q_old = 0.499 * np.ones((n_subj))
# alpha = 0.1 * np.ones((n_subj))
# for reward, choice in zip(rewards, left_choices):
#     print("Q_old:", Q_old)
#     print("Choice:", choice)
#     print("Reward:", reward)
#     Q_old = update_Q(reward, choice, Q_old, alpha)

# Fit model
with pm.Model() as model:

    # # Population-level priors (as un-informative as possible)
    # alpha_mu = pm.Uniform('alpha_mu', lower=0, upper=1)
    # beta_mu = pm.Lognormal('beta_mu', mu=0, sd=1)
    # epsilon_mu = pm.Uniform('epsilon_mu', lower=0, upper=1)
    #
    # alpha_sd = T.as_tensor_variable(0.1)  # pm.HalfNormal('alpha_sd', sd=0.5)
    # beta_sd = T.as_tensor_variable(3)  # pm.HalfNormal('beta_sd', sd=3)
    # epsilon_sd = T.as_tensor_variable(0.1)  # pm.HalfNormal('epsilon_sd', sd=0.5)
    #
    # # Individual parameters - hierarchical (specify bounds to avoid initial energy==-inf because logp(RV)==-inf)
    # alpha = pm.Bound(pm.Normal, lower=0, upper=1)('alpha', mu=alpha_mu, sd=alpha_sd, shape=n_subj)
    # beta = pm.Bound(pm.Normal, lower=0)('beta', mu=beta_mu, sd=beta_sd, shape=n_subj)
    # epsilon = pm.Bound(pm.Normal, lower=0, upper=1)('epsilon', mu=epsilon_mu, sd=epsilon_sd, shape=n_subj)

    # Individual parameters - flat
    alpha = pm.Uniform('alpha', lower=0, upper=1, shape=n_subj)
    beta = pm.Lognormal('beta', mu=0, sd=1, shape=n_subj)
    epsilon = pm.Uniform('epsilon', lower=0, upper=1, shape=n_subj)

    # Observed data (choices & rewards)
    rewards = T.as_tensor_variable(rewards)
    left_choices = T.as_tensor_variable(left_choices)
    right_choices = T.as_tensor_variable(right_choices)

    # Calculate Q-values using RL model
    Q_old = T.as_tensor_variable(0.49 * np.ones(n_subj))
    Q_left, _ = theano.scan(fn=update_Q,
                            sequences=[rewards, left_choices],
                            outputs_info=[Q_old],
                            non_sequences=[alpha])
    Q_right, _ = theano.scan(fn=update_Q,
                             sequences=[rewards, right_choices],
                             outputs_info=[Q_old],
                             non_sequences=[alpha])

    # Get probabilities (one of the two actions is enough; add initial Q as first element in trial 0)
    p_left = p_from_Q(Q_left, Q_right, beta, epsilon)[:-1]  # remove last entry
    Q_old = T.as_tensor_variable(0.49 * np.ones((1, n_subj)))  # get first entry
    complete_p_left = T.concatenate([Q_old, p_left], axis=0)  # combine

    # Use Bernoulli to sample responses
    model_choices = pm.Bernoulli('model_choices', p=complete_p_left, observed=left_choices)

    if verbose:

        # Check model logp and RV logps (will crash if they are nan or -inf)
        print("model.logp(model.test_point):", model.logp(model.test_point))
        for RV in model.basic_RVs:
            print("logp of {0}: {1}:".format(RV.name, RV.logp(model.test_point)))

        # Print variable start_values for debugging
        Q_old_print = T.printing.Print('Q_old')(Q_old)
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

    # Sample the model (should be >=5000, tune>=500)
    trace = pm.sample(10000, tune=500, chains=3, cores=1)

    # Show results
    pm.traceplot(trace)
    model_summary = pm.summary(trace)

    # Save results
    now = datetime.datetime.now()
    model_id = str(now.year) + '_' + str(now.month) + '_' + str(now.day) + '_nsubj' + str(n_subj) + 'Flat_mu'
    with open(base_dir + 'trace_' + model_id + '.pickle', 'wb') as handle:
        pickle.dump(trace, handle)
    with open(base_dir + 'model_' + model_id + '.pickle', 'wb') as handle:
        pickle.dump(model, handle)
    plt.savefig(base_dir + model_id + 'plot.png')
