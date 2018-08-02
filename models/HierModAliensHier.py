import pickle

import theano
theano.exception_verbosity = 'high'
import matplotlib.pyplot as plt

from shared_modeling_simulation import *
from modeling_helpers import *


# Switches for this script
run_on_cluster = False
verbose = False
print_logps = False
file_name_suff = 'test_hier'
use_fake_data = False

# Which data should be fitted?
fitted_data_name = 'humans'  # 'humans', 'simulations'

# Sampling details
n_samples = 50
n_tune = 10
max_n_subj = 20
if run_on_cluster:
    n_cores = 3
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

# Print data
if verbose:
    print('seasons {0}'.format(seasons))
    print('aliens {0}'.format(aliens))
    print('actions {0}'.format(actions))
    print('rewards {0}'.format(rewards))

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

    # Sample subject parameters
    alpha = pm.Beta('alpha', alpha=2, beta=2, shape=n_subj, testval=np.random.choice([0.1, 0.5], n_subj))
    beta = pm.Gamma('beta', alpha=7.5, beta=1.0, shape=n_subj, testval=5 * np.random.rand(n_subj).round(2))
    if verbose:
        T.printing.Print('beta')(beta)
        T.printing.Print('alpha')(alpha)

    # Initialize Q-values
    Q_low0 = alien_initial_Q * T.ones([n_subj, n_TS, n_aliens, n_actions])
    Q_high0 = alien_initial_Q * T.ones([n_subj, n_seasons, n_TS])
    if verbose:
        T.printing.Print('Q_low trial 0')(Q_low0)
        T.printing.Print('Q_high trial 0')(Q_high0)

    # Define function to update Q-values based on stimulus, action, and reward
    def update_Q_low(season, alien, action, reward, Q_low, alpha):
        # Loop over trials: take data for all subjects, 1 single trial
        Q_low_new = Q_low.copy()
        RPE_low = alpha * (reward - Q_low_new[T.arange(n_subj), season, alien, action])

        if verbose:
            T.printing.Print('Q_low inside update_Q_low')(Q_low_new)
            T.printing.Print('alien inside update_Q_low')(alien)
            T.printing.Print('action inside update_Q_low')(action)
            T.printing.Print('reward inside update_Q_low')(reward)
            T.printing.Print('RPE_low')(RPE_low)
        Q_low_new = T.set_subtensor(Q_low_new[T.arange(n_subj), season, alien, action],
                                    Q_low_new[T.arange(n_subj), season, alien, action] + RPE_low)
        return Q_low_new

    # Get Q-values for the whole task (pdate each trial)
    Q_low, _ = theano.scan(fn=update_Q_low,
                       sequences=[seasons, aliens, actions, rewards],
                       outputs_info=[Q_low0],
                       non_sequences=[alpha])

    Q_low = T.concatenate([[Q_low0], Q_low[:-1]], axis=0)  # Add first trial's Q-values, remove last trials Q-values
    if verbose:
        T.printing.Print('Q_low all subj, all trials')(Q_low)

    # Select the right Q-values for each trial
    betas = T.tile(T.repeat(beta, n_actions), n_trials).reshape([n_trials, n_subj, n_actions])
    Q_sub = Q_low[trials, subj, seasons, aliens]
    if verbose:
        T.printing.Print('betas * Q_sub')(betas * Q_sub)
        T.printing.Print('(betas * Q_sub).shape')((betas * Q_sub).shape)

    # First step to translate into probabilities: apply exp
    p = T.exp(betas * Q_sub)
    if verbose:
        T.printing.Print('p')(p)
        T.printing.Print('p.shape')(p.shape)

    # Bring p in the correct shape for pm.Categorical: 2-d array of trials
    action_wise_p = p.reshape([n_trials * n_subj, n_actions])
    action_wise_actions = actions.flatten()
    if verbose:
        T.printing.Print('action_wise_p.shape')(action_wise_p.shape)
        T.printing.Print('action_wise_p')(action_wise_p)
        T.printing.Print('flat actions.shape')(action_wise_actions.shape)
        T.printing.Print('flat actions')(action_wise_actions)

    # Select actions (& second step of calculating probabilities: normalize)
    actions = pm.Categorical('actions', p=action_wise_p, observed=action_wise_actions)

    # Check model logp and RV logps (will crash if they are nan or -inf)
    if verbose or print_logps:
        print_logp_info(model)

    # Sample the model
    trace = pm.sample(n_samples, tune=n_tune, chains=n_chains, cores=n_cores)

# Get results
model.name = 'aliens'
model_dict.update({model: trace})
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
