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
# p = eps / n_actions + (1 - eps) * softmax(Q)
# requirements file
# issues.md -> all the issues and how I solved them


# Switches for this script
run_on_cluster = False
print_logps = False
file_name_suff = 'h'
use_fake_data = False

# Which data should be fitted?
fitted_data_name = 'humans'  # 'humans', 'simulations'

# Sampling details
n_samples = 50
n_tune = 50
max_n_subj = 1  # set > 31 to include all subjects
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

# trials, subj = np.meshgrid(range(n_trials), range(n_subj))
# trials = T.as_tensor_variable(trials.T)
# subj = T.as_tensor_variable(subj.T)

# Get save directory and identifier
save_dir, save_id = get_save_dir_and_save_id(run_on_cluster, file_name_suff, fitted_data_name, n_samples)

# Fit models
model_dict = {}
print("Compiling models for {0} with {1} samples and {2} tuning steps...\n".format(fitted_data_name, n_samples, n_tune))
with pm.Model() as model:

    ## RL parameters: softmax temperature beta; learning rate alpha; forgetting of Q-values
    # Parameter means
    beta = pm.Gamma('beta', mu=1, sd=2, testval=1.5)
    alpha = pm.Uniform('alpha', lower=0, upper=1, testval=0.1)
    forget = pm.Uniform('forget', lower=0, upper=1, testval=0.001)
    beta_high = pm.Gamma('beta_high', mu=1, sd=2, testval=1.5)
    alpha_high = pm.Uniform('alpha_high', lower=0, upper=1, testval=0.1)
    forget_high = pm.Uniform('forget_high', lower=0, upper=1, testval=0.001)

    # Print resulting parameters
    T.printing.Print('beta')(beta)
    T.printing.Print('alpha')(alpha)
    T.printing.Print('forget')(forget)
    T.printing.Print('beta_high')(beta_high)
    T.printing.Print('alpha_high')(alpha_high)
    T.printing.Print('forget_high')(forget_high)

    ## Select action based on Q-values
    # Initialize Q-values
    Q_low0 = alien_initial_Q * T.ones([n_TS, n_aliens, n_actions])
    Q_high0 = alien_initial_Q * T.ones([n_seasons, n_TS])

    # Calculate Q-values (update in each trial)
    [Q_low, Q_high, TS], _ = theano.scan(fn=update_Qs_1subj,
                                         sequences=[seasons, aliens, actions, rewards],
                                         outputs_info=[Q_low0, Q_high0, None],
                                         non_sequences=[beta_high, alpha, alpha_high, forget, forget_high, n_subj])

    Q_low = T.concatenate([[Q_low0], Q_low[:-1]], axis=0)  # Add first trial's Q-values, remove last trials Q-values
    Q_high = T.concatenate([[Q_high0], Q_high[:-1]], axis=0)  # Won't be used - just for printing
    T.printing.Print("Q_low")(Q_low)
    T.printing.Print("Q_high")(Q_high)

    # Select Q-values for each trial & translate into probabilities
    Q_sub = Q_low[T.arange(n_trials), TS.flatten(), aliens.flatten()]  # Q_sub.shape -> [n_trials, n_subj, n_actions]
    Q_sub = beta * Q_sub
    p_low = T.nnet.softmax(Q_sub)

    # Select actions based on Q-values
    action_wise_actions = actions.flatten()
    actions = pm.Categorical('actions', p=p_low, observed=action_wise_actions)
    T.printing.Print('Q_sub')(Q_sub)
    T.printing.Print('p_low')(p_low)

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
