import pickle

import pymc3 as pm
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from shared_aliens import *
from modeling_helpers import *

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


# Switches for this script
run_on_cluster = False
print_logps = False
file_name_suff = 'f'
use_fake_data = False

# Which data should be fitted?
fitted_data_name = 'humans'  # 'humans', 'simulations'

# Sampling details
n_samples = 50
n_tune = 50
max_n_subj = 20  # set > 31 to include all subjects
if run_on_cluster:
    n_cores = 4
else:
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
    # n_subj, n_trials, seasons, aliens, actions, rewards =\
    #     load_aliens_data(run_on_cluster, fitted_data_name, max_n_subj, verbose)

    # pd.DataFrame(seasons).to_csv("seasons.csv", index=False)
    # pd.DataFrame(aliens).to_csv("aliens.csv", index=False)
    # pd.DataFrame(actions).to_csv("actions.csv", index=False)
    # pd.DataFrame(rewards).to_csv("rewards.csv", index=False)
    seasons = pd.read_csv("../notebooks/1735_data/seasons.csv")
    aliens = pd.read_csv("../notebooks/1735_data/aliens.csv")
    actions = pd.read_csv("../notebooks/1735_data/actions.csv")
    rewards = pd.read_csv("../notebooks/1735_data/rewards.csv")
    n_trials = seasons.shape[0]
    n_subj = seasons.shape[1]

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

    ## RL parameters: softmax temperature beta; learning rate alpha; forgetting of Q-values
    # Parameter shapes
    beta_shape = (1, n_subj, 1)  # Q_sub.shape -> [n_trials, n_subj, n_actions]
    forget_shape = (n_subj, 1, 1, 1)  # Q_low[0].shape -> [n_subj, n_TS, n_aliens, n_actions]
    beta_high_shape = (n_subj, 1)  # -> [n_subj, n_TS]
    forget_high_shape = (n_subj, 1, 1)  # -> [n_subj, n_seasons, n_TS]

    # Parameter means
    beta_mu = pm.Gamma('beta_mu', mu=1, sd=2, testval=1.5)
    alpha_mu = pm.Uniform('alpha_mu', lower=0, upper=1, testval=0.2)
    forget_mu = pm.Uniform('forget_mu', lower=0, upper=1, testval=0.1)
    # alpha_mu = pm.Bound(pm.HalfNormal, upper=1)('alpha_mu', sd=0.5, testval=0.2)
    # forget_mu = pm.Bound(pm.HalfNormal, upper=1)('forget_mu', sd=0.5, testval=0.1)
    beta_high_mu = pm.Gamma('beta_high_mu', mu=1, sd=2, testval=1.5)
    alpha_high_mu = pm.Bound(pm.HalfNormal, upper=1)('alpha_high_mu', sd=0.5, testval=0.2)
    forget_high_mu = pm.Bound(pm.HalfNormal, upper=1)('forget_high_mu', sd=0.5, testval=0.1)

    # Parameter sds
    beta_sd = pm.HalfNormal('beta_sd', sd=1, testval=0.1)
    alpha_sd = pm.HalfNormal('alpha_sd', sd=0.2, testval=0.1)
    forget_sd = pm.HalfNormal('forget_sd', sd=0.2, testval=0.1)
    beta_high_sd = pm.HalfNormal('beta_high_sd', sd=1, testval=0.1)
    alpha_high_sd = pm.HalfNormal('alpha_high_sd', sd=0.2, testval=0.1)
    forget_high_sd = pm.HalfNormal('forget_high_sd', sd=0.2, testval=0.1)

    # Individual differences
    beta_matt = pm.Bound(pm.Normal, lower=-beta_mu / beta_sd)(
        'beta_matt', mu=0, sd=1,
        shape=beta_shape, testval=np.random.choice([-1, 0, 1], n_subj).reshape(beta_shape))
    alpha_matt = pm.Bound(pm.Normal, lower=-alpha_mu / alpha_sd, upper=(1 - alpha_mu) / alpha_sd)(
        'alpha_matt', mu=0, sd=1,
        shape=n_subj, testval=np.random.choice([-1, 0, 1], n_subj))
    forget_matt = pm.Bound(pm.Normal, lower=-forget_mu / forget_sd, upper=(1 - forget_mu) / forget_sd)(
        'forget_matt', mu=0, sd=1,
        shape=forget_shape, testval=np.random.choice([0, 1], n_subj).reshape(forget_shape))
    beta_high_matt = pm.Bound(pm.Normal, lower=-beta_high_mu / beta_high_sd)(
        'beta_high_matt', mu=0, sd=1,
        shape=beta_high_shape, testval=np.random.choice([-1, 0, 1], n_subj).reshape(beta_high_shape))
    alpha_high_matt = pm.Bound(pm.Normal, lower=-alpha_high_mu / alpha_high_sd, upper=(1 - alpha_high_mu) / alpha_high_sd)(
        'alpha_high_matt', mu=0, sd=1,
        shape=n_subj, testval=np.random.choice([-1, 0, 1], n_subj))
    forget_high_matt = pm.Bound(pm.Normal, lower=-forget_high_mu / forget_high_sd, upper=(1 - forget_high_mu) / forget_high_sd)(
        'forget_high_matt', mu=0, sd=1,
        shape=forget_high_shape, testval=np.random.choice([0, 1], n_subj).reshape(forget_high_shape))

    # Put parameters together
    beta = pm.Deterministic('beta', beta_mu + beta_sd * beta_matt)
    alpha = pm.Deterministic('alpha', alpha_mu + alpha_sd * alpha_matt)
    forget = pm.Deterministic('forget', forget_mu + forget_sd * forget_matt)
    beta_high = pm.Deterministic('beta_high', beta_high_mu + beta_high_sd * beta_high_matt)
    alpha_high = pm.Deterministic('alpha_high', alpha_high_mu + alpha_high_sd * alpha_high_matt)
    forget_high = pm.Deterministic('forget_high', forget_high_mu + forget_high_sd * forget_high_matt)
    # beta_high = beta.dimshuffle(1, 2)  # [n_trials, n_subj, n_actions] -> [n_subj, n_TS]
    # alpha_high = alpha.copy()  # [n_subj]
    # forget_high = forget.dimshuffle(0, 2, 3)  # [n_subj, n_TS, n_aliens, n_actions] -> [n_subj, n_seasons, n_TS]

    # Print resulting parameters
    T.printing.Print('beta')(beta)
    T.printing.Print('alpha')(alpha)
    T.printing.Print('forget')(forget)
    T.printing.Print('beta_high')(beta_high)
    T.printing.Print('alpha_high')(alpha_high)
    T.printing.Print('forget_high')(forget_high)

    ## Select action based on Q-values
    # Initialize Q-values
    Q_low0 = alien_initial_Q * T.ones([n_subj, n_TS, n_aliens, n_actions])
    Q_high0 = alien_initial_Q * T.ones([n_subj, n_seasons, n_TS])

    # Calculate Q-values (update in each trial)
    [Q_low, Q_high, TS], _ = theano.scan(fn=update_Qs,
                                         sequences=[seasons, aliens, actions, rewards],
                                         outputs_info=[Q_low0, Q_high0, None],
                                         non_sequences=[beta_high, alpha, alpha_high, forget, forget_high, n_subj])

    Q_low = T.concatenate([[Q_low0], Q_low[:-1]], axis=0)  # Add first trial's Q-values, remove last trials Q-values
    Q_high = T.concatenate([[Q_high0], Q_high[:-1]], axis=0)  # Won't be used - just for printing
    T.printing.Print("Q_low")(Q_low)
    T.printing.Print("Q_high")(Q_high)

    # Select Q-values for each trial & translate into probabilities
    Q_sub = beta * Q_low[trials, subj, TS, aliens]  # Q_sub.shape -> [n_trials, n_subj, n_actions]
    action_wise_Q = Q_sub.reshape([n_trials * n_subj, n_actions])
    action_wise_p = T.nnet.softmax(action_wise_Q)

    # Select actions based on Q-values
    action_wise_actions = actions.flatten()
    actions = pm.Categorical('actions', p=action_wise_p, observed=action_wise_actions)
    T.printing.Print('action_wise_Q')(action_wise_Q)
    T.printing.Print('action_wise_p')(action_wise_p)

    # Check logps and draw samples
    # print_logp_info(model)
    trace = pm.sample(n_samples, tune=n_tune, chains=n_chains, cores=n_cores, nuts_kwargs=dict(target_accept=.80))

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
