import pickle

import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from shared_modeling_simulation import *
from modeling_helpers import *


# Switches for this script
run_on_cluster = False
verbose = False
print_logps = False
file_name_suff = 'hier_abf'
use_fake_data = False

# Which data should be fitted?
fitted_data_name = 'humans'  # 'humans', 'simulations'

# Sampling details
n_samples = 200
n_tune = 50
max_n_subj = 100  # set > 31 to include all subjects
upper = 10
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

    # Priors (crashes if testvals are not specified!)
    beta_mu_mu = pm.Uniform('beta_mu_mu', lower=0, upper=5, testval=1)
    beta_mu_sd = pm.Uniform('beta_mu_sd', lower=0, upper=5, testval=0.5)
    beta_sd_mu = pm.Uniform('beta_sd_mu', lower=0, upper=5, testval=1)
    beta_sd_sd = pm.Uniform('beta_sd_sd', lower=0, upper=5, testval=0.5)
    beta_mu = pm.Gamma('beta_mu', mu=beta_mu_mu, sd=beta_mu_sd, testval=1)
    beta_sd = pm.Gamma('beta_sd', mu=beta_sd_mu, sd=beta_sd_sd, testval=0.5)

    alpha_mu_mu = pm.Uniform('alpha_mu_mu', lower=0, upper=1, testval=0.2)
    alpha_mu_sd = pm.Uniform('alpha_mu_sd', lower=0, upper=1, testval=0.1)
    alpha_sd_mu = pm.Uniform('alpha_sd_mu', lower=0, upper=1, testval=0.2)
    alpha_sd_sd = pm.Uniform('alpha_sd_sd', lower=0, upper=1, testval=0.1)
    alpha_mu = pm.Beta('alpha_mu', mu=alpha_mu_mu, sd=alpha_mu_sd, testval=0.2)
    alpha_sd = pm.Beta('alpha_sd', mu=alpha_sd_mu, sd=alpha_sd_sd, testval=0.1)

    forget_mu_mu = pm.Uniform('forget_mu_mu', lower=0, upper=1, testval=0.2)
    forget_mu_sd = pm.Uniform('forget_mu_sd', lower=0, upper=1, testval=0.1)
    forget_sd_mu = pm.Uniform('forget_sd_mu', lower=0, upper=1, testval=0.2)
    forget_sd_sd = pm.Uniform('forget_sd_sd', lower=0, upper=1, testval=0.1)
    forget_mu = pm.Beta('forget_mu', mu=forget_mu_mu, sd=forget_mu_sd, testval=0.2)
    forget_sd = pm.Beta('forget_sd', mu=forget_sd_mu, sd=forget_sd_sd, testval=0.1)

    # Sample subject parameters
    beta = pm.Gamma('beta', mu=beta_mu, sd=beta_sd, shape=n_subj, testval=np.random.choice([0.8, 1.2], n_subj))
    beta_high = beta.copy()
    alpha = pm.Beta('alpha', mu=alpha_mu, sd=alpha_sd, shape=n_subj, testval=np.random.choice([0.1, 0.5], n_subj))
    alpha_high = alpha.copy()
    forget = pm.Beta('forget', mu=forget_mu, sd=forget_sd, shape=n_subj, testval=np.random.choice([0.1, 0.5], n_subj))
    # forget = T.zeros_like(alpha)
    forget_high = forget.copy()

    # Adjust shapes for manipulations later-on
    beta_lows = T.tile(T.repeat(beta, n_actions), n_trials).reshape([n_trials, n_subj, n_actions])    # Q_sub.shape
    beta_highs = T.repeat(beta_high, n_TS).reshape([n_subj, n_TS])  # Q_high_sub.shape

    forget_lows = T.repeat(forget, n_TS * n_aliens * n_actions).reshape([n_subj, n_TS, n_aliens, n_actions])  # Q_low for 1 trial
    forget_highs = T.repeat(forget_high, n_seasons * n_TS).reshape([n_subj, n_seasons, n_TS])  # Q_high for 1 trial

    if verbose:
        T.printing.Print('beta')(beta)
        T.printing.Print('alpha')(alpha)
        T.printing.Print('forget')(forget)
        T.printing.Print('beta_high')(beta_high)
        T.printing.Print('alpha_high')(alpha_high)
        T.printing.Print('forget_high')(forget_high)

    # Initialize Q-values
    Q_low0 = alien_initial_Q * T.ones([n_subj, n_TS, n_aliens, n_actions])
    Q_high0 = alien_initial_Q * T.ones([n_subj, n_seasons, n_TS])
    if verbose:
        T.printing.Print('Q_low trial 0')(Q_low0)
        T.printing.Print('Q_high trial 0')(Q_high0)

    # Define function to update Q-values based on stimulus, action, and reward
    def update_Qs(season, alien, action, reward, Q_low, Q_high, alpha_low, alpha_high, beta_high, forget_low, forget_high):
        # Loop over trials: take data for all subjects, 1 single trial

        # Select TS
        Q_high_sub = Q_high[T.arange(n_subj), season]
        p_high = T.exp(beta_high * Q_high_sub)
        TS = p_high.argmax(axis=1)  # Select TS deterministically
        # TS = pm.Categorical('TS', p=p_high, shape=n_subj, testval=np.ones(n_subj), observed=np.random.choice(n_subj))  # Select probabilistically

        # Forget Q-values a little bit
        Q_low_new = (1 - forget_low) * Q_low
        Q_high_new = (1 - forget_high) * Q_high

        # Calculate RPEs & update Q-values
        RPE_low = alpha_low * (reward - Q_low_new[T.arange(n_subj), TS, alien, action])
        Q_low_new = T.set_subtensor(Q_low_new[T.arange(n_subj), TS, alien, action],
                                    Q_low_new[T.arange(n_subj), TS, alien, action] + RPE_low)

        RPE_high = alpha_high * (reward - Q_high_new[T.arange(n_subj), season, TS])
        Q_high_new = T.set_subtensor(Q_high_new[T.arange(n_subj), season, TS],
                                     Q_high_new[T.arange(n_subj), season, TS] + RPE_high)

        if verbose:
            T.printing.Print('Q_low.shape')(Q_low.shape)
            T.printing.Print('forget.shape')(forget.shape)
            T.printing.Print('Q_low inside update_Qs')(Q_low_new)
            T.printing.Print('alien inside update_Qs')(alien)
            T.printing.Print('action inside update_Qs')(action)
            T.printing.Print('reward inside update_Qs')(reward)
            T.printing.Print('RPE_low')(RPE_low)
            T.printing.Print('TS')(TS)
            T.printing.Print('Q_high_sub.shape')(Q_high_sub.shape)
            T.printing.Print('Q_high_sub')(Q_high_sub)
            T.printing.Print('beta_high.shape')(beta_high.shape)
            T.printing.Print('p_high.shape')(p_high.shape)
            T.printing.Print('p_high')(p_high)

        return [Q_low_new, Q_high_new]

    # Get Q-values for the whole task (update each trial)
    [Q_low, Q_high], _ = theano.scan(fn=update_Qs,
                                     sequences=[seasons, aliens, actions, rewards],
                                     outputs_info=[Q_low0, Q_high0],
                                     non_sequences=[alpha, alpha_high, beta_highs, forget_lows, forget_highs])

    Q_low = T.concatenate([[Q_low0], Q_low[:-1]], axis=0)  # Add first trial's Q-values, remove last trials Q-values
    if verbose:
        T.printing.Print('Q_low all subj, all trials')(Q_low)

    # Select the right Q-values for each trial
    Q_sub = Q_low[trials, subj, seasons, aliens]
    if verbose:
        T.printing.Print('Q_subj.shape')(Q_sub.shape)
        T.printing.Print('Q_subj')(Q_sub)
        T.printing.Print('beta_lows * Q_sub')(beta_lows * Q_sub)
        T.printing.Print('(beta_lows * Q_sub).shape')((beta_lows * Q_sub).shape)

    # First step to translate into probabilities: apply exp
    p = T.exp(beta_lows * Q_sub)
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
    # pd.DataFrame(pm.compare(model_dict).to_csv(save_dir + save_id + model.name + 'waic.csv'))

# Save results
print('Saving trace, trace plot, model, and model summary to {0}{1}...\n'.format(save_dir, save_id + model.name))
with open(save_dir + save_id + model.name + '.pickle', 'wb') as handle:
    pickle.dump({'trace': trace, 'model': model, 'summary': model_summary},
                handle, protocol=pickle.HIGHEST_PROTOCOL)
