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
file_name_suff = 'test'

# Which data should be fitted?
fitted_data_name = 'humans'  # 'humans', 'simulations'

# Sampling details
n_samples = 1000
n_tune = 100
if run_on_cluster:
    n_cores = 3
else:
    n_cores = 1
n_chains = n_cores

# Load to-be-fitted data
n_subj, seasons, aliens, actions, rewards = load_aliens_data(run_on_cluster, fitted_data_name, verbose)
subj_range = range(n_subj)

# Prepare things for saving
save_dir, save_id = get_save_dir_and_save_id(run_on_cluster, file_name_suff, fitted_data_name, n_samples)

# Fit models
model_dict = {}
print("Compiling models for {0} with {1} samples and {2} tuning steps...\n".format(fitted_data_name, n_samples, n_tune))

# Fake data
# p = np.ones([n_trials, n_subj, n_stimuli, n_actions])
# p[:, :, 0, :] = [1, 0, 0]  # all subj always choose action 0 for stimulus 0
# p[:, :, 1, :] = [0, 1, 0]  # all subj always choose action 1 for stimulus 1
# p[:, :, 2, :] = [0, 0, 1]  # all subj always choose action 2 for stimulus 2
# p = T.as_tensor_variable(p)
#
# stim = np.ones([n_trials, n_subj], dtype=int)
# stim.astype(int)
# stim = T.as_tensor_variable(stim)

n_trials, n_subj, n_stimuli, n_actions = 5, 2, 4, 3
# actions = np.array([
#     np.random.choice(range(n_actions), size=n_trials, p=[.1, .1, .8]),
#     np.random.choice(range(n_actions), size=n_trials, p=[.1, .8, .1]),
#     np.random.choice(range(n_actions), size=n_trials, p=[.8, .1, .1])
#     ]).T
# actions = T.as_tensor_variable(actions)
stimuli = np.random.choice(range(n_stimuli), size=[n_trials, n_subj])
actions = np.random.choice(range(n_actions), size=[n_trials, n_subj])
rewards = 10 * np.random.rand(n_trials * n_subj).reshape([n_trials, n_subj]).round(2)
print('actions {0}'.format(actions))
print('stimuli {0}'.format(stimuli))
print('rewards {0}'.format(rewards))
actions = T.as_tensor_variable(actions)
stimuli = T.as_tensor_variable(stimuli)
rewards = T.as_tensor_variable(rewards)

with pm.Model() as model:

    # One alpha, one beta per subject
    alpha = pm.Uniform('alpha', lower=0, upper=1, shape=n_subj, testval=np.random.choice([0.1, 0.5], n_subj))
    beta = pm.HalfNormal('beta', sd=5, shape=n_subj, testval=5 * np.random.rand(n_subj).round(2))
    T.printing.Print('beta')(beta)
    T.printing.Print('alpha')(alpha)

    # Initialize Q-values
    # Q = 10 * np.random.rand(n_subj * n_stimuli * n_actions).reshape([n_subj, n_stimuli, n_actions]).round(2)
    # Q = T.as_tensor_variable(Q)
    Q0 = alien_initial_Q * T.ones([n_subj, n_stimuli, n_actions])
    # T.printing.Print('Q trial 0')(Q0)

    def update_Q(stimulus, action, reward, Q, alpha):
        # Loop over trials: take data for all subjects, 1 single trial
        Q_new = Q.copy()
        # T.printing.Print('Q inside update_Q')(Q_new)
        # T.printing.Print('stimulus inside update_Q')(stimulus)
        # T.printing.Print('action inside update_Q')(action)
        # T.printing.Print('reward inside update_Q')(reward)
        # T.printing.Print('Q[T.arange(n_subj), stimulus, action]')(Q[T.arange(n_subj), stimulus, action])
        RPE = alpha * (reward - Q_new[T.arange(n_subj), stimulus, action])
        # T.printing.Print('RPE')(RPE)
        Q_new = T.set_subtensor(Q_new[T.arange(n_subj), stimulus, action], RPE)
        # T.printing.Print('Q_new')(Q_new)
        return Q_new

    Q, _ = theano.scan(fn=update_Q,
                       sequences=[stimuli, actions, rewards],
                       outputs_info=[Q0],
                       non_sequences=[alpha])
    Q = T.concatenate([[Q0], Q[:-1]], axis=0)  # Add first trial's Q-values, remove last trials Q-values
    T.printing.Print('Q all subj, all trials')(Q)

    # Transform Q into p
    def softmax(Q, stimulus, beta):
        # Loop over subjects within one trial
        # T.printing.Print('Q inside softmax')(Q)
        # T.printing.Print('stimulus inside softmax')(stimulus)
        # T.printing.Print('beta inside softmax')(beta)
        Q_stim = Q[stimulus]
        # T.printing.Print('Q_stim')(Q_stim)
        Q_exp = T.exp(beta * Q_stim)
        # T.printing.Print('Q_exp')(Q_exp)
        p = Q_exp / T.sum(Q_exp)
        return p

    def softmax_wrapper(Q, stimulus, beta):
        # Loop over trials
        p, _ = theano.scan(fn=softmax,
                           sequences=[Q, stimulus, beta])
        # T.printing.Print('p for all subj, 1 trial')(p)
        return p

    p, _ = theano.scan(fn=softmax_wrapper,
                       sequences=[Q, stimuli],
                       non_sequences=[beta])

    T.printing.Print('p all subj, all trials')(p)
    action_wise_p = p.flatten().reshape([n_trials * n_subj, n_actions])
    T.printing.Print('action_wise_p')(action_wise_p)
    T.printing.Print('flat actions')(actions.flatten())

    actions = pm.Categorical('actions', p=action_wise_p, observed=actions.flatten())  #, observed=T.ones([3, 10])

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
