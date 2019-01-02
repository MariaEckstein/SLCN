import pymc3 as pm
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd

from shared_aliens import alien_initial_Q, update_Qs
from modeling_helpers import load_aliens_data, get_save_dir_and_save_id, plot_gen_rec

# THINGS
# simulation = pm.sample_ppc(trace, samples=500)
# plt.hist(pm.Bound(pm.Normal, lower=0).dist(mu=1, sd=3).random(size=int(1e6)), bins=int(1e2)); plt.show()

# HOW TO
# http://deeplearning.net/software/theano/extending/graphstructures.html#apply
# theano.printing.pydotprint()  # TypeError: pydotprint() missing 1 required positional argument: 'fct'
# figure out theano.config.floatX==float32 (http://deeplearning.net/software/theano/library/config.html)
# debug theano: http://deeplearning.net/software/theano/cifarSC2011/advanced_theano.html#debugging

# Switches
verbose = False
run_on_cluster = False
print_logps = False
file_name_suff = 'hier'
param_names = ['alpha', 'beta', 'forget', 'alpha_high', 'beta_high', 'forget_high']  # Don't change
use_fake_data = True

# Which data should be fitted?
fitted_data_name = 'humans'  # 'humans', 'simulations'
n_seasons, n_TS, n_aliens, n_actions = 3, 3, 4, 3

# Sampling details
max_n_subj = 3  # set to 31 to include all subjects
max_n_trials = 440  # 440 / 960
if run_on_cluster:
    n_cores = 4
    n_chains = 2
    n_samples = 100
    n_tune = 100
else:  # debug version
    n_cores = 1
    n_samples = 10
    n_tune = 5
    n_chains = n_cores

if use_fake_data:
    n_subj, n_trials = 2, 5
    seasons = np.ones([n_trials, n_subj], dtype=int)  # np.random.choice(range(n_seasons), size=[n_trials, n_subj])
    aliens = np.ones([n_trials, n_subj], dtype=int)  # np.random.choice(range(n_aliens), size=[n_trials, n_subj])
    actions = np.ones([n_trials, n_subj], dtype=int)  # np.random.choice(range(n_actions), size=[n_trials, n_subj])
    rewards = 10 * np.ones([n_trials, n_subj])  # np.random.rand(n_trials * n_subj).reshape([n_trials, n_subj]).round(2)
else:
    n_subj, n_trials, seasons, aliens, actions, rewards, true_params = \
        load_aliens_data(run_on_cluster, fitted_data_name, param_names, file_name_suff, max_n_subj, max_n_trials,
                         verbose)

    # pd.DataFrame(seasons).to_csv("seasons.csv", index=False)
    # pd.DataFrame(aliens).to_csv("aliens.csv", index=False)
    # pd.DataFrame(actions).to_csv("actions.csv", index=False)
    # pd.DataFrame(rewards).to_csv("rewards.csv", index=False)
    # pd.DataFrame(true_params).to_csv("true_params.csv")
    # seasons = pd.read_csv("seasons.csv")
    # aliens = pd.read_csv("aliens.csv")
    # actions = pd.read_csv("actions.csv")
    # rewards = pd.read_csv("rewards.csv")
    # n_trials = seasons.shape[0]
    # n_subj = seasons.shape[1]

if 'fs' in file_name_suff:  # 'flat-stimulus' model: completely ignores the context
    seasons = np.zeros(seasons.shape, dtype=int)

trials, subj = np.meshgrid(range(n_trials), range(n_subj))
trials = T.as_tensor_variable(trials.T)
subj = T.as_tensor_variable(subj.T)

# Convert data to tensor variables
seasons = theano.shared(np.asarray(seasons, dtype='int32'))
aliens = theano.shared(np.asarray(aliens, dtype='int32'))
actions = theano.shared(np.asarray(actions, dtype='int32'))
rewards = theano.shared(np.asarray(rewards, dtype='int32'))

# Get save directory and identifier
save_dir, save_id = get_save_dir_and_save_id(run_on_cluster, file_name_suff, fitted_data_name, n_samples)

# Fit models
print("Compiling model: {4} {0}, {1} samples, {2} tuning steps, {3} trials\n".
      format(fitted_data_name, n_samples, n_tune, max_n_trials, max_n_subj))

with pm.Model() as model:

    # RL parameters: softmax temperature beta; learning rate alpha; forgetting of Q-values
    beta_shape = (n_subj, 1)  # Q_sub.shape inside scan -> [n_subj, n_actions]
    forget_shape = (n_subj, 1, 1, 1)  # Q_low.shape inside scan -> [n_subj, n_TS, n_aliens, n_actions]
    beta_high_shape = (n_subj, 1)  # Q_high_sub.shape inside scan -> [n_subj, n_TS]
    forget_high_shape = (n_subj, 1, 1)  # Q_high.shape inside scan -> [n_subj, n_seasons, n_TS]

    alpha_mu = pm.HalfNormal('alpha_mu', sd=0.1)
    alpha_sd = pm.HalfNormal('alpha_sd', sd=0.1)
    beta_mu = pm.Bound(pm.Normal, lower=0)('beta_mu', mu=1, sd=2)
    beta_sd = pm.HalfNormal('beta_sd', sd=1)
    forget_mu = pm.HalfNormal('forget_mu', sd=0.05)
    forget_sd = pm.HalfNormal('forget_sd', sd=0.05)

    alpha_high_mu = pm.HalfNormal('alpha_high_mu', sd=0.1)
    alpha_high_sd = pm.HalfNormal('alpha_high_sd', sd=0.1)
    beta_high_mu = pm.Bound(pm.Normal, lower=0)('beta_high_mu', mu=1, sd=2)
    beta_high_sd = pm.HalfNormal('beta_high_sd', sd=1)
    forget_high_mu = pm.HalfNormal('forget_high_mu', sd=0.05)
    forget_high_sd = pm.HalfNormal('forget_high_sd', sd=0.05)

    alpha = pm.Beta('alpha', mu=alpha_mu, sd=alpha_sd, shape=n_subj)
    beta = pm.Bound(pm.Normal, lower=0)('beta', mu=beta_mu, sd=beta_sd, shape=beta_shape)
    forget = pm.Beta('forget', mu=forget_mu, sd=forget_sd, shape=forget_shape)
    # alpha = pm.HalfNormal('alpha', sd=0.1, shape=n_subj)
    # beta = pm.Bound(pm.Normal, lower=0)('beta', mu=1, sd=2, shape=beta_shape)
    # forget = pm.HalfNormal('forget', sd=0.05, shape=forget_shape)

    # alpha_high = pm.Deterministic('alpha_high', alpha.copy())
    # beta_high = pm.Deterministic('beta_high', beta.flatten().reshape(beta_shape))
    # forget_high = pm.Deterministic('forget_high', forget.flatten().reshape(forget_high_shape))
    alpha_high = pm.Beta('alpha_high', mu=alpha_high_mu, sd=alpha_high_sd, shape=n_subj)
    beta_high = pm.Bound(pm.Normal, lower=0)('beta_high', mu=beta_high_mu, sd=beta_high_sd, shape=beta_shape)
    forget_high = pm.Beta('forget_high', mu=forget_high_mu, sd=forget_high_sd, shape=forget_high_shape)
    # alpha_high = pm.HalfNormal('alpha_high', sd=0.1, shape=n_subj)
    # beta_high = pm.Bound(pm.Normal, lower=0)('beta_high', mu=1, sd=2, shape=beta_high_shape)
    # forget_high = pm.HalfNormal('forget_high', sd=0.05, shape=forget_high_shape)
    # alpha_high = pm.Deterministic('alpha_high', 0.2 * T.ones(n_subj))  # Flat agent
    # beta_high = pm.Deterministic('beta_high', 2 * T.ones(beta_high_shape))
    # forget_high = pm.Deterministic('forget_high', T.zeros(forget_high_shape))

    # Calculate Q_high and Q_low for each trial
    Q_low0 = alien_initial_Q * T.ones([n_subj, n_TS, n_aliens, n_actions])
    Q_high0 = alien_initial_Q * T.ones([n_subj, n_seasons, n_TS])
    [Q_low, _, TS, p_low], _ = theano.scan(fn=update_Qs,
                                           sequences=[seasons, aliens, actions, rewards],
                                           outputs_info=[Q_low0, Q_high0, None, None],
                                           non_sequences=[beta, beta_high, alpha, alpha_high, forget, forget_high, n_subj, n_TS])

    # Relate calculated p_low to observed actions
    actions = pm.Categorical('actions',
                             p=p_low.reshape([n_trials * n_subj, n_actions]),
                             observed=actions.flatten())

#     # Draw samples
#     map_estimate = pm.find_MAP()
#
# map_gen_rec = true_params.append(pd.DataFrame([map_estimate[param_name].flatten() for param_name in param_names], index=param_names))
# map_gen_rec.to_csv(save_dir + save_id + '_map_gen_rec.csv')
# if not run_on_cluster:
#     plot_gen_rec(param_names=param_names, gen_rec=map_gen_rec, save_name=save_dir + save_id + '_map_gen_rec_plot.png')
#
# with model:
    MCMC_trace = pm.sample(n_samples, tune=n_tune, chains=n_chains, cores=n_cores)  #, start=map_estimate

print("WAIC: {0}".format(pm.waic(MCMC_trace, model).WAIC))
MCMC_model_summary = pm.summary(MCMC_trace)
pd.DataFrame(MCMC_model_summary).to_csv(save_dir + save_id + '_summary.csv')
mcmc_params = np.full((len(param_names), n_subj), np.nan)
for i, param_name in enumerate(param_names):
    idxs = MCMC_model_summary.index.str.contains(param_name + '__')
    mcmc_params[i] = np.array(MCMC_model_summary.loc[idxs, 'mean'])
mcmc_params = pd.DataFrame(mcmc_params, index=param_names)
mcmc_gen_rec = true_params.append(mcmc_params)
mcmc_gen_rec.to_csv(save_dir + save_id + '_mcmc_gen_rec.csv')

if not run_on_cluster:
    pm.traceplot(MCMC_trace)
    plt.savefig(save_dir + save_id + '_traceplot.png')
    plot_gen_rec(param_names=param_names, gen_rec=mcmc_gen_rec, save_name=save_dir + save_id + '_mcmc_gen_rec_plot.png')

# Save results
print('Saving trace, trace plot, model, and model summary to {0}{1}...\n'.format(save_dir, save_id + model.name))
with open(save_dir + save_id + model.name + '.pickle', 'wb') as handle:
    pickle.dump({'trace': MCMC_trace, 'model': model, 'summary': MCMC_model_summary},
                handle, protocol=pickle.HIGHEST_PROTOCOL)
