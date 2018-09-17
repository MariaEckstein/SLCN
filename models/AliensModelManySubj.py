import pymc3 as pm
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd

from shared_aliens import alien_initial_Q, update_Qs_flat, update_Qs_hier  #, update_Qs_hier
from modeling_helpers import load_aliens_data, get_save_dir_and_save_id

# TODO
# sample TS rather than argmax
# independent low & high parameters
# more efficient?!?
# respond to pymc3discourse
# simulate
# figure out which model makes good simulations and implement this one
#
# backend = pymc3.backends.sqlite.SQLite('aliens_trace')
# sample from posterior predictive distribution
# simulation = pm.sample_ppc(trace, samples=500)
# p = eps / n_actions + (1 - eps) * softmax(Q)
# requirements file
# issues.md -> all the issues and how I solved them
# to plot distributions: plt.hist(pm.Bound(pm.Normal, lower=0).dist(mu=1, sd=3).random(size=int(1e6)), bins=int(1e2)); plt.show()

# How to stuff
# http://deeplearning.net/software/theano/extending/graphstructures.html#apply
# theano.printing.pydotprint()  # TypeError: pydotprint() missing 1 required positional argument: 'fct'
# figure out theano.config.floatX==float32 (http://deeplearning.net/software/theano/library/config.html)
# debug theano: http://deeplearning.net/software/theano/cifarSC2011/advanced_theano.html#debugging

# ISSUES
# Exception: ('Compilation failed (return status=1): C:\\Users\\maria\\Anaconda3\\envs\\PYMC3\\libs/python36.lib: error adding symbols: File in wrong format\r. collect2.exe: error: ld returned 1 exit status\r. ', '[Shape(v)]') -> conda install mingw libpython

# Switches for this script
verbose = False
run_on_cluster = False
print_logps = False
file_name_suff = 'h_many'
use_fake_data = False

# Which data should be fitted?
fitted_data_name = 'humans'  # 'humans', 'simulations'
n_seasons, n_TS, n_aliens, n_actions = 3, 3, 4, 3

# Sampling details
max_n_subj = 30  # set > 31 to include all subjects
max_n_trials = 20
if run_on_cluster:
    n_cores = 4
    n_samples = 2000
    n_tune = 2000
else:
    n_cores = 1
    n_samples = 10
    n_tune = 10
n_chains = 1

if use_fake_data:
    n_subj, n_trials = 2, 5
    seasons = np.ones([n_trials, n_subj], dtype=int)  # np.random.choice(range(n_seasons), size=[n_trials, n_subj])
    aliens = np.ones([n_trials, n_subj], dtype=int)  # np.random.choice(range(n_aliens), size=[n_trials, n_subj])
    actions = np.ones([n_trials, n_subj], dtype=int)  # np.random.choice(range(n_actions), size=[n_trials, n_subj])
    rewards = 10 * np.ones([n_trials, n_subj])  # np.random.rand(n_trials * n_subj).reshape([n_trials, n_subj]).round(2)
else:
    n_subj, n_trials, seasons, aliens, actions, rewards =\
        load_aliens_data(run_on_cluster, fitted_data_name, max_n_subj, max_n_trials, verbose)

    # pd.DataFrame(seasons).to_csv("seasons.csv", index=False)
    # pd.DataFrame(aliens).to_csv("aliens.csv", index=False)
    # pd.DataFrame(actions).to_csv("actions.csv", index=False)
    # pd.DataFrame(rewards).to_csv("rewards.csv", index=False)
    # seasons = pd.read_csv("seasons.csv")
    # aliens = pd.read_csv("aliens.csv")
    # actions = pd.read_csv("actions.csv")
    # rewards = pd.read_csv("rewards.csv")
    # n_trials = seasons.shape[0]
    # n_subj = seasons.shape[1]

if 'fs' in file_name_suff:
    seasons = np.zeros(seasons.shape, dtype=int)

# Convert data to tensor variables
seasons = theano.shared(np.asarray(seasons, dtype='int32'))
aliens = theano.shared(np.asarray(aliens, dtype='int32'))
actions = theano.shared(np.asarray(actions, dtype='int32'))
rewards = theano.shared(np.asarray(rewards, dtype='int32'))

trials, subj = np.meshgrid(range(n_trials), range(n_subj))
trials = T.as_tensor_variable(trials.T)
subj = T.as_tensor_variable(subj.T)

# Get save directory and identifier
save_dir, save_id = get_save_dir_and_save_id(run_on_cluster, file_name_suff, fitted_data_name, n_samples)

# Fit models
model_dict = {}
print("Compiling models for {4} {0} with {1} samples, {2} tuning steps, and {3} trials...\n".
      format(fitted_data_name, n_samples, n_tune, max_n_trials, max_n_subj))

# RL MODEL
# with pm.Model() as model:
#
#     ## RL parameters: softmax temperature beta; learning rate alpha; forgetting of Q-values
#     beta_shape = (1, n_subj, 1)  # Q_sub.shape -> [n_trials, n_subj, n_actions]
#     forget_shape = (n_subj, 1, 1, 1)  # Q_low[0].shape -> [n_subj, n_TS, n_aliens, n_actions]
#
#     beta = pm.Bound(pm.Normal, lower=0)('beta', mu=1, sd=5, shape=beta_shape, testval=1.5 * T.ones(beta_shape))
#     alpha = pm.Uniform('alpha', lower=0, upper=1, shape=n_subj, testval=0.1 * T.ones(n_subj))
#     forget = pm.Uniform('forget', lower=0, upper=1, shape=forget_shape, testval=0.001 * T.ones(forget_shape))
#
#     ## Select action based on Q-values
#     Q_low0 = alien_initial_Q * T.ones([n_subj, n_TS, n_aliens, n_actions])
#     Q_low, _ = theano.scan(fn=update_Qs_flat,
#                            sequences=[seasons, aliens, actions, rewards],
#                            outputs_info=[Q_low0],
#                            non_sequences=[alpha, forget, n_subj],
#                            profile=True,
#                            name='my_scan')
#
#     Q_low = T.concatenate([[Q_low0], Q_low[:-1]], axis=0)  # Add first trial's Q-values, remove last trials Q-values
#
#     # Select Q-values for each trial & translate into probabilities
#     Q_sub = beta * Q_low[trials, subj, seasons, aliens]  # Q_sub.shape -> [n_trials, n_subj, n_actions]
#     p_low = T.nnet.softmax(Q_sub.reshape([n_trials * n_subj, n_actions]))
#
#     # Select actions based on Q-values
#     actions = pm.Categorical('actions', p=p_low, observed=actions.flatten())
#
#     # Draw samples
#     trace = pm.sample(n_samples, tune=n_tune, chains=n_chains, cores=n_cores)

# TS MODEL
with pm.Model() as model:

    ## RL parameters: softmax temperature beta; learning rate alpha; forgetting of Q-values
    beta_shape = (1, n_subj, 1)  # Q_sub.shape -> [n_trials, n_subj, n_actions]
    forget_shape = (n_subj, 1, 1, 1)  # Q_low[0].shape -> [n_subj, n_TS, n_aliens, n_actions]

    beta = pm.Bound(pm.Normal, lower=0)('beta', mu=1, sd=5, shape=beta_shape, testval=1.5 * T.ones(beta_shape))
    alpha = pm.Uniform('alpha', lower=0, upper=1, shape=n_subj, testval=0.1 * T.ones(n_subj))
    forget = pm.Uniform('forget', lower=0, upper=1, shape=forget_shape, testval=0.001 * T.ones(forget_shape))
    alpha_high = pm.Uniform('alpha_high', lower=0, upper=1, shape=n_subj, testval=0.1)

    ## Select action based on Q-values
    Q_low0 = alien_initial_Q * T.ones([n_subj, n_TS, n_aliens, n_actions])
    Q_high0 = alien_initial_Q * T.ones([n_subj, n_seasons, n_TS])
    [Q_low, _, TS], _ = theano.scan(fn=update_Qs_hier,
                           sequences=[seasons, aliens, actions, rewards],
                           outputs_info=[Q_low0, Q_high0, None],
                           non_sequences=[alpha, alpha_high, forget, n_subj],
                           profile=True,
                           name='my_scan')

    Q_low = T.concatenate([[Q_low0], Q_low[:-1]], axis=0)  # Add first trial's Q-values, remove last trials Q-values

    # Select Q-values for each trial & translate into probabilities
    Q_sub = beta * Q_low[trials, subj, TS, aliens]  # Q_sub.shape -> [n_trials, n_subj, n_actions]
    p_low = T.nnet.softmax(Q_sub.reshape([n_trials * n_subj, n_actions]))

    # Select actions based on Q-values
    actions = pm.Categorical('actions', p=p_low, observed=actions.flatten())

    # Draw samples
    trace = pm.sample(n_samples, tune=n_tune, chains=n_chains, cores=n_cores)

model.profile(model.logpt).summary()

# Get results
model_summary = pm.summary(trace)

if not run_on_cluster:
    pm.traceplot(trace)
    plt.savefig(save_dir + save_id + model.name + 'trace_plot.png')
    pd.DataFrame(model_summary).to_csv(save_dir + save_id + model.name + 'model_summary.csv')

# Save results
print('Saving trace, trace plot, model, and model summary to {0}{1}...\n'.format(save_dir, save_id + model.name))
with open(save_dir + save_id + model.name + '.pickle', 'wb') as handle:
    pickle.dump({'trace': trace, 'model': model, 'summary': model_summary},
                handle, protocol=pickle.HIGHEST_PROTOCOL)

# with pm.Model() as hier_model:
#
#     ## RL parameters: softmax temperature beta; learning rate alpha; forgetting of Q-values
#     # Parameter means
#     beta = pm.Bound(pm.Normal, lower=0)('beta', mu=1, sd=5, testval=1.5)
#     alpha = pm.Uniform('alpha', lower=0, upper=1, testval=0.1)
#     forget = pm.Uniform('forget', lower=0, upper=1, testval=0.001)
#     alpha_high = pm.Uniform('alpha_high', lower=0, upper=1, testval=0.1)
#
#     ## Select action based on Q-values
#     Q_low0 = alien_initial_Q * T.ones([n_TS, n_aliens, n_actions])
#     Q_high0 = alien_initial_Q * T.ones([n_seasons, n_TS])
#
#     [Q_low, _, TS], _ = theano.scan(fn=update_Qs_1subj_hier,
#                                        sequences=[seasons, aliens, actions, rewards],
#                                        outputs_info=[Q_low0, Q_high0, None],
#                                        non_sequences=[alpha, alpha_high, forget],
#                                   profile=True,
#                                   name='my_scan')
#
#     Q_low = T.concatenate([[Q_low0], Q_low[:-1]], axis=0)  # Add first trial's Q-values, remove last trials Q-values
#
#     # Select Q-values for each trial & translate into probabilities
#     Q_sub = Q_low[T.arange(n_trials), TS.flatten(), aliens.flatten()]  # Q_sub.shape -> [n_trials, n_subj, n_actions]
#     Q_sub = beta * Q_sub
#     p_low = T.nnet.softmax(Q_sub)
#
#     # Select actions based on Q-values
#     action_wise_actions = actions.flatten()
#     actions = pm.Categorical('actions', p=p_low, observed=action_wise_actions)
#
#     # Check logps and draw samples
#     hier_trace = pm.sample(n_samples, tune=n_tune, chains=n_chains, cores=n_cores, nuts_kwargs=dict(target_accept=.80))

# hier_model.profile(hier_model.logpt).summary()
#
# plt.hist(trace['alpha'])
# plt.show()
# plt.hist(trace['beta'])
# plt.show()
# plt.hist(trace['forget'])
# plt.show()
#
# # Get results
# model_summary = pm.summary(trace)
#
# if not run_on_cluster:
#     pm.traceplot(trace)
#     plt.savefig(save_dir + save_id + model.name + 'trace_plot.png')
#     pd.DataFrame(model_summary).to_csv(save_dir + save_id + model.name + 'model_summary.csv')
#
# # Save results
# print('Saving trace, trace plot, model, and model summary to {0}{1}...\n'.format(save_dir, save_id + model.name))
# with open(save_dir + save_id + model.name + '.pickle', 'wb') as handle:
#     pickle.dump({'trace': trace, 'model': model, 'summary': model_summary},
#                 handle, protocol=pickle.HIGHEST_PROTOCOL)
