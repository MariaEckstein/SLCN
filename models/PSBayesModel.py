import pickle
import numpy as np

import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from shared_modeling_simulation import get_likelihoods, post_from_lik
from modeling_helpers import load_data, get_save_dir_and_save_id, print_logp_info
import pymc3 as pm


# Switches for this script
run_on_cluster = False
verbose = True
file_name_suff = 'betswirew'

upper = 1000

# Which data should be fitted?
fitted_data_name = 'humans'  # 'humans', 'simulations'
kids_and_teens_only = False
adults_only = False

# Sampling details
n_samples = 50
n_tune = 10
n_cores = 1
n_chains = 1
target_accept = 0.8

# Load to-be-fitted data
n_subj, rewards, choices, group, n_groups = load_data(run_on_cluster, fitted_data_name, kids_and_teens_only, adults_only, verbose)

rewards = theano.shared(np.asarray(rewards, dtype='int32'))
choices = theano.shared(np.asarray(choices, dtype='int32'))
group = theano.shared(np.asarray(group, dtype='int32'))

# Prepare things for saving
save_dir, save_id = get_save_dir_and_save_id(run_on_cluster, file_name_suff, fitted_data_name, n_samples)

# Fit models
model_dict = {}
print("Compiling models for {0} with {1} samples and {2} tuning steps...\n".format(fitted_data_name, n_samples, n_tune))

with pm.Model() as model:

    # Get population-level and individual parameters
    eps = T.as_tensor_variable(0)
    p_noisy = 1e-5 * T.ones(n_subj)  # TODO: 1e-5 * T.as_tensor_variable(1)

    beta_a_a = pm.Uniform('beta_a_a', lower=0, upper=upper)
    beta_a_b = pm.Uniform('beta_a_b', lower=0, upper=upper)
    beta_b_a = pm.Uniform('beta_b_a', lower=0, upper=upper)
    beta_b_b = pm.Uniform('beta_b_b', lower=0, upper=upper)
    beta_a = pm.Gamma('beta_a', alpha=beta_a_a, beta=beta_a_b, shape=n_groups)
    beta_b = pm.Gamma('beta_b', alpha=beta_b_a, beta=beta_b_b, shape=n_groups)

    p_switch_a_a = pm.Uniform('p_switch_a_a', lower=0, upper=upper)
    p_switch_a_b = pm.Uniform('p_switch_a_b', lower=0, upper=upper)
    p_switch_b_a = pm.Uniform('p_switch_b_a', lower=0, upper=upper)
    p_switch_b_b = pm.Uniform('p_switch_b_b', lower=0, upper=upper)
    p_switch_a = pm.Gamma('p_switch_a', alpha=p_switch_a_a, beta=p_switch_a_b, shape=n_groups)
    p_switch_b = pm.Gamma('p_switch_b', alpha=p_switch_b_a, beta=p_switch_b_b, shape=n_groups)

    p_reward_a_a = pm.Uniform('p_reward_a_a', lower=0, upper=upper)
    p_reward_a_b = pm.Uniform('p_reward_a_b', lower=0, upper=upper)
    p_reward_b_a = pm.Uniform('p_reward_b_a', lower=0, upper=upper)
    p_reward_b_b = pm.Uniform('p_reward_b_b', lower=0, upper=upper)
    p_reward_a = pm.Gamma('p_reward_a', alpha=p_reward_a_a, beta=p_reward_a_b, shape=n_groups)
    p_reward_b = pm.Gamma('p_reward_b', alpha=p_reward_b_a, beta=p_reward_b_b, shape=n_groups)

    beta = pm.Gamma('beta', alpha=beta_a[group], beta=beta_b[group], shape=n_subj)
    p_switch = pm.Beta('p_switch', alpha=p_switch_a[group], beta=p_switch_b[group], shape=n_subj)
    p_reward = pm.Beta('p_reward', alpha=p_reward_a[group], beta=p_reward_b[group], shape=n_subj)

    # Parameter mu and var and group differences
    beta_mu = pm.Deterministic('beta_mu', beta_a / beta_b)
    beta_var = pm.Deterministic('beta_var', beta_a / np.square(beta_b))
    p_switch_mu = pm.Deterministic('p_switch_mu', p_switch_a / p_switch_b)
    p_switch_var = pm.Deterministic('p_switch_var', p_switch_a / np.square(p_switch_b))
    p_reward_mu = pm.Deterministic('p_reward_mu', p_reward_a / p_reward_b)
    p_reward_var = pm.Deterministic('p_reward_var', p_reward_a / np.square(p_reward_b))

    beta_mu_diff01 = pm.Deterministic('beta_mu_diff01', beta_a[0] - beta_a[1])
    beta_mu_diff02 = pm.Deterministic('beta_mu_diff02', beta_a[0] - beta_a[2])
    beta_mu_diff12 = pm.Deterministic('beta_mu_diff12', beta_a[1] - beta_a[2])
    p_switch_mu_diff01 = pm.Deterministic('p_switch_mu_diff01', p_switch_a[0] - p_switch_a[1])
    p_switch_mu_diff02 = pm.Deterministic('p_switch_mu_diff02', p_switch_a[0] - p_switch_a[2])
    p_switch_mu_diff12 = pm.Deterministic('p_switch_mu_diff12', p_switch_a[1] - p_switch_a[2])
    p_reward_mu_diff01 = pm.Deterministic('p_reward_mu_diff01', p_reward_a[0] - p_reward_a[1])
    p_reward_mu_diff02 = pm.Deterministic('p_reward_mu_diff02', p_reward_a[0] - p_reward_a[2])
    p_reward_mu_diff12 = pm.Deterministic('p_reward_mu_diff12', p_reward_a[1] - p_reward_a[2])

    # Get likelihoods
    lik_cor, lik_inc = get_likelihoods(rewards, choices, p_reward, p_noisy)

    # Get posterior, calculate probability of subsequent trial, add eps noise
    p_right = 0.5 * T.ones(n_subj, dtype='int32')
    p_right, _ = theano.scan(fn=post_from_lik,
                             sequences=[lik_cor, lik_inc],
                             outputs_info=[p_right],
                             non_sequences=[p_switch, eps, beta])

    # Add initial p=0.5 at the beginning of p_right
    initial_p = 0.5 * T.ones((1, n_subj))
    p_right = T.concatenate([initial_p, p_right[:-1]], axis=0)

    # Use Bernoulli to sample responses
    model_choices = pm.Bernoulli('model_choices', p=p_right, observed=choices)

    # Check model logp and RV logps (will crash if they are nan or -inf)
    if verbose:
        print_logp_info(model)

    # Sample the model
    trace = pm.sample(n_samples, tune=n_tune, chains=n_chains, cores=n_cores, nuts_kwargs=dict(target_accept=.8))

# Get results
model_summary = pm.summary(trace)

if not run_on_cluster:
    pm.traceplot(trace)
    plt.savefig(save_dir + 'plot_' + save_id + '.png')

# Save results
print('Saving trace, trace plot, model, and model summary to {0}{1}...\n'.format(save_dir, save_id))
with open(save_dir + save_id + '.pickle', 'wb') as handle:
    pickle.dump({'trace': trace, 'model': model, 'summary': model_summary},
                handle, protocol=pickle.HIGHEST_PROTOCOL)
