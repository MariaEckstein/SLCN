import pymc3 as pm
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd

from shared_aliens import alien_initial_Q, update_Qs, update_Qs_flat, update_Qs_hier  #, update_Qs_hier
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
file_name_suff = 'h'
use_fake_data = False

# Which data should be fitted?
fitted_data_name = 'simulations'  # 'humans', 'simulations'
n_seasons, n_TS, n_aliens, n_actions = 3, 3, 4, 3

# Sampling details
max_n_subj = 30  # set > 31 to include all subjects
max_n_trials = 440  # 960
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
    n_subj, n_trials, seasons, aliens, actions, rewards, true_params =\
        load_aliens_data(run_on_cluster, fitted_data_name, file_name_suff, max_n_subj, max_n_trials, verbose)

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
print("Compiling model: {4} {0}, {1} samples, {2} tuning steps, {3} trials\n".
      format(fitted_data_name, n_samples, n_tune, max_n_trials, max_n_subj))

# Model for TS and RL (comment the right stuff in and out!)
with pm.Model() as model:

    # RL parameters: softmax temperature beta; learning rate alpha; forgetting of Q-values
    beta_shape = (n_subj, 1)  # Q_sub.shape inside scan -> [n_subj, n_actions]
    forget_shape = (n_subj, 1, 1, 1)  # Q_low.shape inside scan -> [n_subj, n_TS, n_aliens, n_actions]
    # beta_high_shape = (n_subj, 1)  # Q_high_sub.shape inside scan -> [n_subj, n_TS]
    # forget_high_shape = (n_subj, 1, 1)  # Q_high.shape inside scan -> [n_subj, n_seasons, n_TS]

    # beta = pm.Bound(pm.Normal, lower=0)('beta', mu=1, sd=10, shape=beta_shape, testval=1.5 * T.ones(beta_shape))
    beta = pm.Deterministic('beta', 2 * T.ones(beta_shape))
    # alpha = pm.Uniform('alpha', lower=0, upper=1, shape=n_subj, testval=0.1 * T.ones(n_subj))
    alpha = pm.Deterministic('alpha', 0.1 * T.ones(n_subj))
    forget = pm.Uniform('forget', lower=0, upper=1, shape=forget_shape, testval=0.01 * T.ones(forget_shape))
    # forget = pm.Deterministic('forget', T.zeros(forget_shape))
    # beta_high = T.ones(beta_high_shape)
    # alpha_high = pm.Uniform('alpha_high', lower=0, upper=1, shape=n_subj, testval=0.1)  # Hierarchical agent
    alpha_high = pm.Deterministic('alpha_high', 0.1 * T.ones(n_subj))  # Flat agent
    # forget_high = T.zeros(forget_high_shape)

    # Calculate Q_high and Q_low for each trial
    Q_low0 = alien_initial_Q * T.ones([n_subj, n_TS, n_aliens, n_actions])
    Q_high0 = alien_initial_Q * T.ones([n_subj, n_seasons, n_TS])
    [Q_low, _, TS, p_low], _ = theano.scan(fn=update_Qs,
                                           sequences=[seasons, aliens, actions, rewards],
                                           outputs_info=[Q_low0, Q_high0, None, None],
                                           non_sequences=[beta, alpha, alpha_high, forget, n_subj],
                                           # profile=True,
                                           # name='my_scan')
                                           )

    # Relate calculated p_low to observed actions
    actions = pm.Categorical('actions',
                             p=p_low.reshape([n_trials * n_subj, n_actions]),
                             observed=actions.flatten())

    # Draw samples
    # trace = pm.sample(n_samples, tune=n_tune, chains=n_chains, cores=n_cores)
    map_estimate = pm.find_MAP()  # default: Broyden–Fletcher–Goldfarb–Shanno (BFGS)

# model.profile(model.logpt).summary()

# Look at MAP estimate for all parameters
print(map_estimate)
param_names = ['alpha', 'beta', 'forget', 'alpha_high']
map = pd.DataFrame([map_estimate[RV].flatten() for RV in param_names], index=param_names)
param_data = map.append(true_params)
param_data.to_csv(save_dir + save_id + model.name + 'map.csv', header=False)

axes = plt.subplot(2, 2, 1)
sns.regplot(param_data.loc['alpha'], param_data.loc['true_alpha'], fit_reg=False)
plt.plot(axes.get_xlim(), axes.get_xlim())
axes = plt.subplot(2, 2, 2)
sns.regplot(param_data.loc['beta'], param_data.loc['true_beta'], fit_reg=False)
plt.plot(axes.get_xlim(), axes.get_xlim())
axes = plt.subplot(2, 2, 3)
sns.regplot(param_data.loc['forget'], param_data.loc['true_forget'], fit_reg=False)
plt.plot(axes.get_xlim(), axes.get_xlim())
plt.subplot(2, 2, 4)
sns.regplot(param_data.loc['alpha'], param_data.loc['beta'], fit_reg=False)
plt.savefig(save_dir + save_id + model.name + 'gen_rec_plot.png')

# Check Q_low in each trial
get_Q_low_at_map = model.fn(outs=Q_low)
print("Q_low at map:\n", get_Q_low_at_map(map_estimate))

# Check p_low in each trial
get_p_low_at_map = model.fn(outs=p_low)
print("p_low at map:\n", get_p_low_at_map(map_estimate))

# Check TS in each trial
get_TS_at_map = model.fn(outs=TS)
print("TS at map:\n", get_TS_at_map(map_estimate))

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
