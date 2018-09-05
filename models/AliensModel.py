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
file_name_suff = 'h_abf_efficient'
use_fake_data = False

# Which data should be fitted?
fitted_data_name = 'humans'  # 'humans', 'simulations'

# Sampling details
n_samples = 50
n_tune = 50
max_n_subj = 2  # set > 31 to include all subjects
n_cores = 1
n_chains = 1

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

    pd.DataFrame(seasons).to_csv("seasons.csv", index=False)
    pd.DataFrame(aliens).to_csv("aliens.csv", index=False)
    pd.DataFrame(actions).to_csv("actions.csv", index=False)
    pd.DataFrame(rewards).to_csv("rewards.csv", index=False)
    # seasons = pd.read_csv("seasons.csv")
    # aliens = pd.read_csv("aliens.csv")
    # actions = pd.read_csv("actions.csv")
    # rewards = pd.read_csv("rewards.csv")
    # n_trials = seasons.shape[0]
    # n_subj = seasons.shape[1]

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

    # RL parameters: softmax temperature beta; learning rate alpha; forgetting of Q-values
    beta_shape = (1, n_subj, 1)  # Q_sub.shape -> [n_trials, n_subj, n_actions]
    beta_mu = pm.Gamma('beta_mu', mu=1, sd=2, testval=1.25)
    beta_sd = pm.HalfNormal('beta_sd', sd=0.5, testval=0.1)
    beta_matt = pm.Bound(pm.Normal, lower=-beta_mu / beta_sd)(
        'beta_matt', mu=0, sd=1,
        shape=beta_shape, testval=np.random.choice([-1, 0, 1], n_subj).reshape(beta_shape))
    beta = pm.Deterministic('beta', beta_mu + beta_sd * beta_matt)
    T.printing.Print('beta')(beta)

    alpha_mu = pm.Uniform('alpha_mu', lower=0, upper=1, testval=0.15)
    alpha_sd = pm.HalfNormal('alpha_sd', sd=0.1, testval=0.05)
    alpha_matt = pm.Bound(pm.Normal, lower=-alpha_mu / alpha_sd, upper=(1 - alpha_mu) / alpha_sd)(
        'alpha_matt', mu=0, sd=1,
        shape=n_subj, testval=np.random.choice([-1, 0, 1], n_subj))
    alpha = pm.Deterministic('alpha', alpha_mu + alpha_sd * alpha_matt)
    T.printing.Print('alpha')(alpha)

    forget_shape = (n_subj, 1, 1, 1)  # Q_low[0].shape -> [n_subj, n_TS, n_aliens, n_actions]
    forget_mu = pm.HalfNormal('forget_mu', sd=0.1, testval=0.04)
    forget_sd = pm.HalfNormal('forget_sd', sd=0.1, testval=0.01)
    forget_matt = pm.Bound(pm.Normal, lower=-forget_mu / forget_sd, upper=(1 - forget_mu) / forget_sd)(
        'forget_matt', mu=0, sd=1,
        shape=forget_shape, testval=np.random.choice([-1, 0, 1], n_subj).reshape(forget_shape))
    forget = pm.Deterministic('forget', forget_mu + forget_sd * forget_matt)
    T.printing.Print('forget')(forget)

    beta_high = beta.dimshuffle(1, 2)  # [n_trials, n_subj, n_actions] -> [n_subj, n_TS]
    alpha_high = alpha.copy()  # [n_subj]
    forget_high = forget.dimshuffle(0, 2, 3)  # [n_subj, n_TS, n_aliens, n_actions] -> [n_subj, n_seasons, n_TS]

    # Initialize Q-values
    Q_low0 = alien_initial_Q * T.ones([n_subj, n_TS, n_aliens, n_actions])
    Q_high0 = alien_initial_Q * T.ones([n_subj, n_seasons, n_TS])

    # Get Q-values for the whole task (update each trial)
    [Q_low, Q_high], _ = theano.scan(fn=update_Qs,  # hierarchical: TS = p_high.argmax(axis=1); flat: TS = season
                                     sequences=[seasons, aliens, actions, rewards],
                                     outputs_info=[Q_low0, Q_high0],
                                     non_sequences=[beta_high, alpha, alpha_high, forget, forget_high, n_subj])

    Q_low = T.concatenate([[Q_low0], Q_low[:-1]], axis=0)  # Add first trial's Q-values, remove last trials Q-values
    Q_high = T.concatenate([[Q_high0], Q_high[:-1]], axis=0)  # Won't be used - just for printing
    T.printing.Print("Q_low")(Q_low)
    T.printing.Print("Q_high")(Q_high)

    # Select the right Q-values for each trial & apply softmax
    Q_sub = beta * Q_low[trials, subj, seasons, aliens]  # Q_sub.shape -> [n_trials, n_subj, n_actions]
    action_wise_Q = Q_sub.reshape([n_trials * n_subj, n_actions])
    action_wise_p = T.nnet.softmax(action_wise_Q)  # TODO: test!

    # Select action based on Q-values
    action_wise_actions = actions.flatten()
    actions = pm.Categorical('actions', p=action_wise_p, observed=action_wise_actions)
    T.printing.Print('action_wise_Q')(action_wise_Q)
    T.printing.Print('action_wise_p')(action_wise_p)

    # Sample the model
    # TODO: backend = pymc3.backends.sqlite.SQLite('aliens_trace')
    trace = pm.sample(n_samples, tune=n_tune, chains=n_chains, cores=n_cores, nuts_kwargs=dict(target_accept=.80))
    # TODO: sample from posterior predictive distribution
    # simulation = pm.sample_ppc(trace, samples=500)

if verbose:
    plt.hist(trace['alpha'])
    plt.show()
    plt.hist(trace['beta'])
    plt.show()
    plt.hist(trace['forget'])
    plt.show()

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
