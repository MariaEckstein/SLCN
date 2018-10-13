import pymc3 as pm
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd

from AlienTask import Task
from shared_aliens import alien_initial_Q, update_Qs, update_Qs_th_sim
from modeling_helpers import load_aliens_data, get_save_dir_and_save_id, plot_gen_rec

# Switches for this script
verbose = False
run_on_cluster = False
print_logps = False
file_name_suff = 'f_mse_MAP'
param_names = ['alpha', 'beta', 'forget', 'alpha_high', 'beta_high', 'forget_high']
use_fake_data = False

# Which data should be fitted?
fitted_data_name = 'humans'  # 'humans', 'simulations'
n_seasons, n_TS, n_aliens, n_actions = 3, 3, 4, 3

# Sampling details
max_n_subj = 31  # set > 31 to include all subjects
n_sim_per_subj = 2
max_n_trials = 440  # 440 / 960
if run_on_cluster:
    n_cores = 4
    n_chains = 2
    n_samples = 100
    n_tune = 100
else:
    n_cores = 1
    n_samples = 100
    n_tune = 20
    n_chains = n_cores

def_TS = np.array([[[1, 6, 1],  # alien0, items0-2
                    [1, 1, 4],  # alien1, items0-2
                    [5, 1, 1],  # etc.
                    [10, 1, 1]],
                   # TS1
                   [[1, 1, 2],  # alien0, items0-2
                    [1, 8, 1],  # etc.
                    [1, 1, 7],
                    [1, 3, 1]],
                   # TS2
                   [[1, 1, 7],  # TS2
                    [3, 1, 1],
                    [1, 3, 1],
                    [2, 1, 1]]])
if use_fake_data:
    n_subj, n_trials = 2, 5
    seasons = np.ones([n_trials, n_subj], dtype=int)  # np.random.choice(range(n_seasons), size=[n_trials, n_subj])
    aliens = np.ones([n_trials, n_subj], dtype=int)  # np.random.choice(range(n_aliens), size=[n_trials, n_subj])
else:
    n_subj, n_trials, seasons, aliens, _, __, true_params = \
        load_aliens_data(run_on_cluster, fitted_data_name, param_names, file_name_suff, max_n_subj, max_n_trials,
                         verbose)

    task = Task(n_subj)
    _, hum_corrects = task.get_trial_sequence("C:/Users/maria/MEGAsync/Berkeley/TaskSets/Data/version3.1/", n_subj, 1)
    hum_corrects = theano.shared(hum_corrects[:n_trials])

if 'fs' in file_name_suff:
    seasons = np.zeros(seasons.shape, dtype=int)

trials, subj = np.meshgrid(range(n_trials), range(n_subj))
trials = T.as_tensor_variable(trials.T)
subj = T.as_tensor_variable(subj.T)

# Convert data to tensor variables
seasons = theano.shared(np.asarray(seasons, dtype='int32'))
aliens = theano.shared(np.asarray(aliens, dtype='int32'))
def_TS = theano.shared(np.asarray(def_TS, dtype='int32'))

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

    # alpha_mu = pm.HalfNormal('alpha_mu', sd=0.1)
    # alpha_sd = pm.HalfNormal('alpha_sd', sd=0.1)
    # beta_mu = pm.Bound(pm.Normal, lower=0)('beta_mu', mu=1, sd=2)
    # beta_sd = pm.HalfNormal('beta_sd', sd=1)
    # forget_mu = pm.HalfNormal('forget_mu', sd=0.05)
    # forget_sd = pm.HalfNormal('forget_sd', sd=0.05)

    # alpha_high_mu = pm.HalfNormal('alpha_high_mu', sd=0.1)
    # alpha_high_sd = pm.HalfNormal('alpha_high_sd', sd=0.1)
    # beta_high_mu = pm.Bound(pm.Normal, lower=0)('beta_high_mu', mu=1, sd=2)
    # beta_high_sd = pm.HalfNormal('beta_high_sd', sd=1)
    # forget_high_mu = pm.HalfNormal('forget_high_mu', sd=0.05)
    # forget_high_sd = pm.HalfNormal('forget_high_sd', sd=0.05)

    # alpha = pm.Beta('alpha', mu=alpha_mu, sd=alpha_sd, shape=n_subj)
    # beta = pm.Bound(pm.Normal, lower=0)('beta', mu=beta_mu, sd=beta_sd, shape=beta_shape)
    # forget = pm.Beta('forget', mu=forget_mu, sd=forget_sd, shape=forget_shape)
    # alpha = pm.HalfNormal('alpha', sd=0.1, shape=n_subj)
    # beta = pm.Bound(pm.Normal, lower=0)('beta', mu=1, sd=2, shape=beta_shape)
    # forget = pm.HalfNormal('forget', sd=0.05, shape=forget_shape)
    alpha = pm.Uniform('alpha', lower=0, upper=1, shape=n_subj)
    beta = pm.Uniform('beta', lower=0, upper=10, shape=beta_shape)
    forget = pm.Uniform('forget', lower=0, upper=1, shape=forget_shape)

    # alpha_high = pm.Beta('alpha_high', mu=alpha_high_mu, sd=alpha_high_sd, shape=n_subj)
    # beta_high = pm.Bound(pm.Normal, lower=0)('beta_high', mu=beta_high_mu, sd=beta_high_sd, shape=beta_shape)
    # forget_high = pm.Beta('forget_high', mu=forget_high_mu, sd=forget_high_sd, shape=forget_high_shape)
    alpha_high = pm.HalfNormal('alpha_high', sd=0.1, shape=n_subj)
    beta_high = pm.Bound(pm.Normal, lower=0)('beta_high', mu=1, sd=2, shape=beta_high_shape)
    forget_high = pm.HalfNormal('forget_high', sd=0.05, shape=forget_high_shape)
    # alpha_high = pm.Deterministic('alpha_high', 0.2 * T.ones(n_subj))  # Flat agent
    # beta_high = pm.Deterministic('beta_high', 2 * T.ones(beta_high_shape))
    # forget_high = pm.Deterministic('forget_high', T.zeros(forget_high_shape))

    # Calculate Q_high and Q_low for each trial
    Q_low0 = alien_initial_Q * T.ones([n_subj, n_TS, n_aliens, n_actions])
    Q_high0 = alien_initial_Q * T.ones([n_subj, n_seasons, n_TS])
    [Q_low, _, TS, actions, rewards, sim_corrects, p_low], _ = theano.scan(fn=update_Qs_th_sim,
                                           sequences=[seasons, aliens],
                                           outputs_info=[Q_low0, Q_high0, None, None, None, None, None],
                                           non_sequences=[beta, beta_high, alpha, alpha_high, forget, forget_high, n_subj, n_TS, n_actions, def_TS])
    T.printing.Print('actions')(actions)

    # Calculate learning curves
    sim_learning_curve = T.mean(sim_corrects, axis=1)
    hum_learning_curve = T.mean(hum_corrects, axis=1)
    MSE = T.mean((hum_learning_curve - sim_learning_curve) ** 2)

    # Tell PyMC3 that we want to minimize MSE
    pm.Potential('MSE', MSE)

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

if not run_on_cluster:
    pm.traceplot(MCMC_trace)
    plt.savefig(save_dir + save_id + '_traceplot.png')
    # plot_gen_rec(param_names=param_names, gen_rec=mcmc_gen_rec, save_name=save_dir + save_id + '_mcmc_gen_rec_plot.png')

# print("WAIC: {0}".format(pm.waic(MCMC_trace, model).WAIC))
MCMC_model_summary = pm.summary(MCMC_trace)
pd.DataFrame(MCMC_model_summary).to_csv(save_dir + save_id + '_summary.csv')
mcmc_params = np.full((len(param_names), n_subj), np.nan)
for i, param_name in enumerate(param_names):
    idxs = MCMC_model_summary.index.str.contains(param_name + '__')
    mcmc_params[i] = np.array(MCMC_model_summary.loc[idxs, 'mean'])
mcmc_params = pd.DataFrame(mcmc_params, index=param_names)
mcmc_gen_rec = true_params.append(mcmc_params)
mcmc_gen_rec.to_csv(save_dir + save_id + '_mcmc_gen_rec.csv')

# Save results
print('Saving trace, trace plot, model, and model summary to {0}{1}...\n'.format(save_dir, save_id + model.name))
with open(save_dir + save_id + model.name + '.pickle', 'wb') as handle:
    pickle.dump({'trace': MCMC_trace, 'model': model, 'summary': MCMC_model_summary},
                handle, protocol=pickle.HIGHEST_PROTOCOL)
