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
verbose = False
file_name_suff = 'betperswirew'

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
persev_bonus = 2 * choices - 1  # recode as -1 for left and +1 for right
persev_bonus = np.concatenate([np.zeros((1, n_subj)), persev_bonus])  # add 0 bonus for first trial

rewards = theano.shared(np.asarray(rewards, dtype='int32'))
choices = theano.shared(np.asarray(choices, dtype='int32'))
group = theano.shared(np.asarray(group, dtype='int32'))
persev_bonus = theano.shared(np.asarray(persev_bonus, dtype='int32'))

# Prepare things for saving
save_dir, save_id = get_save_dir_and_save_id(run_on_cluster, file_name_suff, fitted_data_name, n_samples)

# Fit models
model_dict = {}
print("Compiling models for {0} with {1} samples and {2} tuning steps...\n".format(fitted_data_name, n_samples, n_tune))

with pm.Model() as model:

    # Get population-level and individual parameters
    eps = T.as_tensor_variable(0)
    p_noisy = 1e-5 * T.as_tensor_variable(1)

    beta_mu_mu = pm.Bound(pm.Normal, lower=0)('beta_mu_mu', mu=1, sd=2)
    beta_mu_sd = pm.HalfNormal('beta_mu_sd', sd=2)
    beta_sd_sd = pm.HalfNormal('beta_sd_sd', sd=1)
    beta_mu = pm.Bound(pm.Normal, lower=0)('beta_mu', mu=beta_mu_mu, sd=beta_mu_sd, shape=n_groups)
    beta_sd = pm.HalfNormal('beta_sd', sd=beta_sd_sd, shape=n_groups)
    beta_offset = pm.Normal('beta_offset', mu=0, sd=1, shape=n_subj)

    persev_mu_mu = pm.Bound(pm.Normal, lower=-1, upper=1)('persev_mu_mu', mu=0, sd=0.5)
    persev_mu_sd = pm.HalfNormal('persev_mu_sd', sd=0.5)
    persev_sd_sd = pm.HalfNormal('persev_sd_sd', sd=0.5)
    persev_mu = pm.Bound(pm.Normal, lower=-1, upper=1)('persev_mu', mu=persev_mu_mu, sd=persev_mu_sd, shape=n_groups)
    persev_sd = pm.HalfNormal('persev_sd', sd=persev_sd_sd, shape=n_groups)
    persev_offset = pm.Normal('persev_offset', mu=0, sd=1, shape=n_subj)

    p_switch_mu_mu = pm.Bound(pm.Normal, lower=0, upper=1)('p_switch_mu_mu', mu=0.1, sd=0.5)
    p_switch_mu_sd = pm.HalfNormal('p_switch_mu_sd', sd=0.5)
    p_switch_sd_sd = pm.HalfNormal('p_switch_sd_sd', sd=0.5)
    p_switch_mu = pm.Bound(pm.Normal, lower=0, upper=1)('p_switch_mu', mu=p_switch_mu_mu, sd=p_switch_mu_sd, shape=n_groups)
    p_switch_sd = pm.HalfNormal('p_switch_sd', sd=p_switch_sd_sd, shape=n_groups)
    p_switch_offset = pm.Normal('p_switch_offset', mu=0, sd=1, shape=n_subj)

    p_reward_mu_mu = pm.Bound(pm.Normal, lower=0, upper=1)('p_reward_mu_mu', mu=0.75, sd=0.5)
    p_reward_mu_sd = pm.HalfNormal('p_reward_mu_sd', sd=0.5)
    p_reward_sd_sd = pm.HalfNormal('p_reward_sd_sd', sd=0.5)
    p_reward_mu = pm.Bound(pm.Normal, lower=0, upper=1)('p_reward_mu', mu=p_reward_mu_mu, sd=p_reward_mu_sd, shape=n_groups)
    p_reward_sd = pm.HalfNormal('p_reward_sd', sd=p_reward_sd_sd, shape=n_groups)
    p_reward_offset = pm.Normal('p_reward_offset', mu=0, sd=1, shape=n_subj)

    # Individual parameters
    beta = pm.Deterministic('beta', beta_mu[group] + beta_offset * beta_sd[group])
    persev = pm.Deterministic('persev', persev_mu[group] + persev_offset * persev_sd[group])
    p_switch = pm.Deterministic('p_switch', p_switch_mu[group] + p_switch_offset * p_switch_sd[group])
    p_reward = pm.Deterministic('p_reward', p_reward_mu[group] + p_reward_offset * p_reward_sd[group])

    scaled_persev_bonus = persev_bonus * persev.reshape((1, n_subj))
    T.printing.Print('choices')(choices)
    T.printing.Print('scaled_persev_bonus')(scaled_persev_bonus)

    # Group differences
    beta_mu_diff01 = pm.Deterministic('beta_mu_diff01', beta_mu[0] - beta_mu[1])
    beta_mu_diff02 = pm.Deterministic('beta_mu_diff02', beta_mu[0] - beta_mu[2])
    beta_mu_diff12 = pm.Deterministic('beta_mu_diff12', beta_mu[1] - beta_mu[2])
    persev_mu_diff01 = pm.Deterministic('persev_mu_diff01', persev_mu[0] - persev_mu[1])
    persev_mu_diff02 = pm.Deterministic('persev_mu_diff02', persev_mu[0] - persev_mu[2])
    persev_mu_diff12 = pm.Deterministic('persev_mu_diff12', persev_mu[1] - persev_mu[2])
    p_switch_mu_diff01 = pm.Deterministic('p_switch_mu_diff01', p_switch_mu[0] - p_switch_mu[1])
    p_switch_mu_diff02 = pm.Deterministic('p_switch_mu_diff02', p_switch_mu[0] - p_switch_mu[2])
    p_switch_mu_diff12 = pm.Deterministic('p_switch_mu_diff12', p_switch_mu[1] - p_switch_mu[2])
    p_reward_mu_diff01 = pm.Deterministic('p_reward_mu_diff01', p_reward_mu[0] - p_reward_mu[1])
    p_reward_mu_diff02 = pm.Deterministic('p_reward_mu_diff02', p_reward_mu[0] - p_reward_mu[2])
    p_reward_mu_diff12 = pm.Deterministic('p_reward_mu_diff12', p_reward_mu[1] - p_reward_mu[2])

    # Get likelihoods
    lik_cor, lik_inc = get_likelihoods(rewards, choices, p_reward, p_noisy)

    # Get posterior, calculate probability of subsequent trial, add eps noise
    p_right = 0.5 * T.ones(n_subj, dtype='int32')
    p_right, _ = theano.scan(fn=post_from_lik,
                             sequences=[lik_cor, lik_inc, scaled_persev_bonus],
                             outputs_info=[p_right],
                             non_sequences=[p_switch, eps, beta])

    # Add initial p=0.5 at the beginning of p_right
    initial_p = 0.5 * T.ones((1, n_subj))
    p_right = T.concatenate([initial_p, p_right[:-1]], axis=0)

    # Use Bernoulli to sample responses
    model_choices = pm.Bernoulli('model_choices', p=p_right, observed=choices)

    # Check model logp and RV logps (will crash if they are nan or -inf)
    # if verbose:
    #     print_logp_info(model)

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
