import pickle

import theano
import theano.tensor as T
# theano.exception_verbosity = 'high'
import matplotlib.pyplot as plt

from shared_modeling_simulation import *
from modeling_helpers import *


# Switches for this script
run_on_cluster = False
verbose = False
print_logps = False
file_name_suff = 'test'
use_fake_data = False

# Which data should be fitted?
fitted_data_name = 'humans'  # 'humans', 'simulations'

# Sampling details
n_samples = 100
n_tune = 10
if run_on_cluster:
    n_cores = 3
else:
    n_cores = 1
n_chains = n_cores

# Get data
n_seasons, n_aliens, n_actions = 3, 4, 3
if use_fake_data:
    n_subj, n_trials = 2, 10
    seasons = np.ones([n_trials, n_subj], dtype=int)  # np.random.choice(range(n_seasons), size=[n_trials, n_subj])
    aliens = np.ones([n_trials, n_subj], dtype=int)  # np.random.choice(range(n_aliens), size=[n_trials, n_subj])
    actions = np.random.choice(range(n_actions), size=[n_trials, n_subj])
    rewards = 10 * np.random.rand(n_trials * n_subj).reshape([n_trials, n_subj]).round(2)
else:
    n_subj, n_trials, seasons, aliens, actions, rewards = load_aliens_data(run_on_cluster, fitted_data_name, verbose)

# Print data
print('seasons {0}'.format(seasons))
print('aliens {0}'.format(aliens))
print('actions {0}'.format(actions))
print('rewards {0}'.format(rewards))

# Convert data to tensor variables
seasons = T.as_tensor_variable(seasons)
aliens = T.as_tensor_variable(aliens)
actions = T.as_tensor_variable(actions)
rewards = T.as_tensor_variable(rewards)

# Get save directory and identifier
save_dir, save_id = get_save_dir_and_save_id(run_on_cluster, file_name_suff, fitted_data_name, n_samples)

# Fit models
model_dict = {}
print("Compiling models for {0} with {1} samples and {2} tuning steps...\n".format(fitted_data_name, n_samples, n_tune))
with pm.Model() as model:

    # Sample subject parameters
    alpha = pm.Uniform('alpha', lower=0, upper=1, shape=n_subj, testval=np.random.choice([0.1, 0.5], n_subj))
    beta = pm.HalfNormal('beta', sd=5, shape=n_subj, testval=5 * np.random.rand(n_subj).round(2))
    T.printing.Print('beta')(beta)
    T.printing.Print('alpha')(alpha)

    # Initialize Q-values
    Q0 = alien_initial_Q * T.ones([n_subj, n_seasons, n_aliens, n_actions])
    if verbose:
        T.printing.Print('Q trial 0')(Q0)

    # Define function to update Q-values based on stimulus, action, and reward
    def update_Q(season, alien, action, reward, Q, alpha):
        # Loop over trials: take data for all subjects, 1 single trial
        Q_new = Q.copy()
        RPE = alpha * (reward - Q_new[T.arange(n_subj), season, alien, action])

        if verbose:
            T.printing.Print('Q inside update_Q')(Q_new)
            T.printing.Print('alien inside update_Q')(alien)
            T.printing.Print('action inside update_Q')(action)
            T.printing.Print('reward inside update_Q')(reward)
            T.printing.Print('Q[T.arange(n_subj), alien, action]')(Q[T.arange(n_subj), alien, action])
            T.printing.Print('RPE')(RPE)
        Q_new = T.set_subtensor(Q_new[T.arange(n_subj), season, alien, action],
                                Q_new[T.arange(n_subj), season, alien, action] + RPE)
        return Q_new

    # Get Q-values for the whole task (update each trial)
    Q, _ = theano.scan(fn=update_Q,
                       sequences=[seasons, aliens, actions, rewards],
                       outputs_info=[Q0],
                       non_sequences=[alpha])

    Q = T.concatenate([[Q0], Q[:-1]], axis=0)  # Add first trial's Q-values, remove last trials Q-values
    if verbose:
        T.printing.Print('Q all subj, all trials')(Q)

    # Define function to transform Q-values into action probabilities
    def softmax(Q, season, alien, beta):
        # Loop over subjects within one trial
        Q_stim = Q[season, alien]
        Q_exp = T.exp(beta * Q_stim)
        p = Q_exp / T.sum(Q_exp)

        if verbose:
            T.printing.Print('Q inside softmax')(Q)
            T.printing.Print('season inside softmax')(season)
            T.printing.Print('alien inside softmax')(alien)
            T.printing.Print('beta inside softmax')(beta)
            T.printing.Print('Q_stim')(Q_stim)
            T.printing.Print('Q_exp')(Q_exp)
            T.printing.Print('p inside softmax')(p)
        return p

    def softmax_trial_wrapper(Q, season, alien, beta):
        # Loop over trials
        p, _ = theano.scan(fn=softmax,
                           sequences=[Q, season, alien, beta])
        # T.printing.Print('p for all subj, 1 trial')(p)
        return p

    # Transform Q-values into action probabilities for all subj, all trials
    p, _ = theano.scan(fn=softmax_trial_wrapper,
                       sequences=[Q, seasons, aliens],
                       non_sequences=[beta])

    action_wise_p = p.flatten().reshape([n_trials * n_subj, n_actions])

    if verbose:
        T.printing.Print('p all subj, all trials')(p)
        T.printing.Print('action_wise_p')(action_wise_p)
        T.printing.Print('flat actions')(actions.flatten())

    # Select actions
    actions = pm.Categorical('actions', p=action_wise_p, observed=actions.flatten())

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
