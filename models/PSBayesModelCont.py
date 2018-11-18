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
file_name_suff = 'bpsr'

upper = 1000

# Which data should be fitted?
fitted_data_name = 'humans'  # 'humans', 'simulations'
kids_and_teens_only = True
adults_only = False

# Sampling details
n_samples = 70
n_tune = 30
n_cores = 1
n_chains = 1
target_accept = 0.8

# Load to-be-fitted data
n_subj, rewards, choices, group, n_groups, age_z = load_data(run_on_cluster, fitted_data_name, kids_and_teens_only, adults_only, verbose)
persev_bonus = 2 * choices - 1  # recode as -1 for choice==0 (right?) and +1 for choice==1 (left?)
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

    beta_a = pm.Uniform('beta_a', lower=0, upper=upper)
    beta_b = pm.Uniform('beta_b', lower=0, upper=upper)
    beta = pm.Gamma('beta', alpha=beta_a, beta=beta_b, shape=n_subj, testval=2 * T.ones(n_subj))

    persev_mu = pm.Normal('persev_mu', mu=0, sd=1)
    persev_sd = pm.HalfNormal('persev_sd', sd=1)
    persev_int = pm.Normal('persev_int', mu=persev_mu, sd=persev_sd, shape=(1, n_subj), testval=0.1 * T.ones((1, n_subj)))

    p_switch_a = pm.Uniform('p_switch_a', lower=0, upper=upper)
    p_switch_b = pm.Uniform('p_switch_b', lower=0, upper=upper)
    p_switch = pm.Beta('p_switch', alpha=p_switch_a, beta=p_switch_b, shape=n_subj, testval=0.1 * T.ones(n_subj))

    p_reward_a = pm.Uniform('p_reward_a', lower=0, upper=upper)
    p_reward_b = pm.Uniform('p_reward_b', lower=0, upper=upper)
    p_reward = pm.Beta('p_reward', alpha=p_reward_a, beta=p_reward_b, shape=n_subj, testval=0.75 * T.ones(n_subj))

    # Parameter mu and slope
    beta_mu = pm.Deterministic('beta_mu', beta_a / beta_b)
    p_switch_mu = pm.Deterministic('p_switch_mu', p_switch_a / (p_switch_a + p_switch_b))
    p_reward_mu = pm.Deterministic('p_reward_mu', p_reward_a / (p_reward_a + p_reward_b))

    # beta_slope = pm.Normal('beta_slope', mu=0, sd=1, testval=0.1)
    persev_slope = pm.Normal('persev_slope', mu=0, sd=1, testval=0.1)
    # p_switch_slope = pm.Normal('p_switch_slope', mu=0, sd=1, testval=-0.1)
    # p_reward_slope = pm.Normal('p_reward_slope', mu=0, sd=1, testval=-0.1)

    # Individual parameters
    # beta = pm.Deterministic('beta', beta_int + beta_slope * age_z)
    persev = pm.Deterministic('persev', persev_int + persev_slope * age_z)
    # p_switch = pm.Deterministic('p_switch', p_switch_int + p_switch_slope * age_z)
    # p_reward = pm.Deterministic('p_reward', p_reward_int + p_reward_slope * age_z)
    scaled_persev_bonus = persev_bonus * persev

    # Get likelihoods
    lik_cor, lik_inc = get_likelihoods(rewards, choices, p_reward, p_noisy)

    # Get posterior & calculate probability of subsequent trial
    p_right = 0.5 * T.ones(n_subj, dtype='int32')
    [p_right, p_choice], _ = theano.scan(fn=post_from_lik,
                                         sequences=[lik_cor, lik_inc, scaled_persev_bonus],
                                         outputs_info=[p_right, None],
                                         non_sequences=[p_switch, eps, beta])

    # Add p=0.5 for the first trial
    initial_p = 0.5 * T.ones((1, n_subj))
    p_choice = T.concatenate([initial_p, p_choice[:-1]], axis=0)

    # Use Bernoulli to sample responses
    model_choices = pm.Bernoulli('model_choices', p=p_choice, observed=choices)

    # Check model logp and RV logps (will crash if they are nan or -inf)
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
