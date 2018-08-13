import pickle

import pymc3 as pm
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from shared_aliens import *
from modeling_helpers import *


# Switches for this script
run_on_cluster = False
print_logps = False
file_name_suff = 'fs_ab'  # flat vs hier models: use update_Q_low vs update_Qs inside theano.scan
use_fake_data = False

# Which data should be fitted?
fitted_data_name = 'humans'  # 'humans', 'simulations'

# Sampling details
n_samples = 300
n_tune = 50
max_n_subj = 100  # set > 31 to include all subjects
if run_on_cluster:
    n_cores = 2
else:
    n_cores = 1
n_chains = n_cores

# Get data
n_seasons, n_TS, n_aliens, n_actions = 3, 3, 4, 3
if use_fake_data:
    n_subj, n_trials = 2, 5
    seasons = np.ones([n_trials, n_subj], dtype=int)  # np.random.choice(range(n_seasons), size=[n_trials, n_subj])
    aliens = np.ones([n_trials, n_subj], dtype=int)  # np.random.choice(range(n_aliens), size=[n_trials, n_subj])
    actions = np.ones([n_trials, n_subj], dtype=int)  # np.random.choice(range(n_actions), size=[n_trials, n_subj])
    rewards = 10 * np.ones([n_trials, n_subj])  # np.random.rand(n_trials * n_subj).reshape([n_trials, n_subj]).round(2)
else:
    n_subj, n_trials, seasons, aliens, actions, rewards =\
        load_aliens_data(run_on_cluster, fitted_data_name, max_n_subj, verbose)

if 'fs' in file_name_suff:
    seasons = np.zeros(seasons.shape, dtype=int)

# Convert data to tensor variables
seasons = T.as_tensor_variable(seasons)
aliens = T.as_tensor_variable(aliens)
actions = T.as_tensor_variable(actions)
rewards = T.as_tensor_variable(rewards)

trials, subj = np.meshgrid(range(n_trials), range(n_subj))
trials = T.as_tensor_variable(trials.T)
subj = T.as_tensor_variable(subj.T)

# Get save directory and identifier
save_dir, save_id = get_save_dir_and_save_id(run_on_cluster, file_name_suff, fitted_data_name, n_samples)

# Fit models
model_dict = {}
print("Compiling models for {0} with {1} samples and {2} tuning steps...\n".format(fitted_data_name, n_samples, n_tune))
with pm.Model() as model:

    # Priors (crashes if testvals are not specified!)
    beta_mu = pm.Uniform('beta_mu', lower=0, upper=5, testval=1)
    beta_sd = pm.Uniform('beta_sd', lower=0, upper=5, testval=1)
    beta_matt = pm.Normal('beta_matt', mu=0, sd=1, shape=n_subj, testval=np.random.choice([0.9, 1.1], n_subj))
    beta = pm.Deterministic('beta', beta_mu + beta_sd * beta_matt)

    alpha_mu = pm.Uniform('alpha_mu', lower=0, upper=1, testval=0.1)
    alpha_sd = pm.Uniform('alpha_sd', lower=0, upper=1, testval=0.1)
    alpha_matt = pm.Normal('alpha_matt', mu=0, sd=1, shape=n_subj, testval=np.random.choice([-0.1, 0, 0.1], n_subj))
    alpha = pm.Deterministic('alpha', alpha_mu + alpha_sd * alpha_matt)

    # forget_mu = pm.Uniform('forget_mu', lower=0, upper=1, testval=0.1)
    # forget_sd = pm.Uniform('forget_sd', lower=0, upper=1, testval=0.1)
    # forget_matt = pm.Normal('forget_matt', mu=0, sd=1, shape=n_subj, testval=np.random.choice([-0.1, 0, 0.1], n_subj))
    # forget = pm.Deterministic('forget', forget_mu + forget_sd * forget_matt)
    forget = T.zeros_like(alpha)

    beta_high = beta.copy()
    alpha_high = alpha.copy()
    forget_high = forget.copy()

    # Adjust shapes for manipulations later-on
    betas = T.tile(T.repeat(beta, n_actions), n_trials).reshape([n_trials, n_subj, n_actions])    # Q_sub.shape
    beta_highs = T.repeat(beta_high, n_TS).reshape([n_subj, n_TS])  # Q_high_sub.shape

    forgets = T.repeat(forget, n_TS * n_aliens * n_actions).reshape([n_subj, n_TS, n_aliens, n_actions])  # Q_low for 1 trial
    forget_highs = T.repeat(forget_high, n_seasons * n_TS).reshape([n_subj, n_seasons, n_TS])  # Q_high for 1 trial

    # Initialize Q-values
    Q_low0 = alien_initial_Q * T.ones([n_subj, n_TS, n_aliens, n_actions])
    Q_high0 = alien_initial_Q * T.ones([n_subj, n_seasons, n_TS])

    # Get Q-values for the whole task (update each trial)
    [Q_low, Q_high], _ = theano.scan(fn=update_Q_low,  # hierarchical model -> update_Qs; flat -> update_Q_low
                                     sequences=[seasons, aliens, actions, rewards],
                                     outputs_info=[Q_low0, Q_high0],
                                     non_sequences=[alpha, alpha_high, beta_highs, forgets, forget_highs, n_subj])

    Q_low = T.concatenate([[Q_low0], Q_low[:-1]], axis=0)  # Add first trial's Q-values, remove last trials Q-values
    T.printing.Print("Q_low")(Q_low)

    # Select the right Q-values for each trial
    Q_sub = Q_low[trials, subj, seasons, aliens]

    # First step to translate into probabilities: apply exp
    p = T.exp(betas * Q_sub)

    # Bring p in the correct shape for pm.Categorical: 2-d array of trials
    action_wise_p = p.reshape([n_trials * n_subj, n_actions])
    action_wise_actions = actions.flatten()

    # Select actions (& second step of calculating probabilities: normalize)
    actions = pm.Categorical('actions', p=action_wise_p, observed=action_wise_actions)

    # Check model logp and RV logps (will crash if they are nan or -inf)
    if verbose or print_logps:
        print_logp_info(model)

    # Sample the model
    trace = pm.sample(n_samples, tune=n_tune, chains=n_chains, cores=n_cores, nuts_kwargs=dict(target_accept=.80))

# Get results
model.name = 'aliens'
model_dict.update({model: trace})
model_summary = pm.summary(trace)

if not run_on_cluster:
    pm.traceplot(trace)
    plt.savefig(save_dir + save_id + model.name + 'trace_plot.png')
    pd.DataFrame(model_summary).to_csv(save_dir + save_id + model.name + 'model_summary.csv')
    # pd.DataFrame(pm.compare(model_dict).to_csv(save_dir + save_id + model.name + 'waic.csv'))

# Save results
print('Saving trace, trace plot, model, and model summary to {0}{1}...\n'.format(save_dir, save_id + model.name))
with open(save_dir + save_id + model.name + '.pickle', 'wb') as handle:
    pickle.dump({'trace': trace, 'model': model, 'summary': model_summary},
                handle, protocol=pickle.HIGHEST_PROTOCOL)
