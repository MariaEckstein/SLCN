import pickle

import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from shared_modeling_simulation import update_Q, p_from_Q
from modeling_helpers import load_data, get_save_dir_and_save_id, print_logp_info

import pymc3 as pm
import numpy as np

# Switches for this script
run_on_cluster = False
verbose = False
print_logps = False
file_name_suff = 'abcnp'
upper = 1000

# Which data should be fitted?
fitted_data_name = 'humans'  # 'humans', 'simulations'
kids_and_teens_only = False
adults_only = False
n_groups = 1

# Sampling details
n_samples = 100
n_tune = 50
n_cores = 2
n_chains = 1
target_accept = 0.8

# Load to-be-fitted data
n_subj, rewards, choices, group, n_groups, age_z = load_data(run_on_cluster, fitted_data_name, n_groups, kids_and_teens_only, adults_only, verbose)

# Create persev_bonus and next_choices
prev_choices = np.concatenate([np.zeros((1, n_subj)), choices])[:-1]  # add 0 in first row, remove last row
prev_choices_inv = np.concatenate([np.zeros((1, n_subj)), 1 - choices])[:-1]  # add 0 in first row, remove last row
# next_choices = np.concatenate([choices[1:], 0.5 * np.ones((1, n_subj))])  # remove first row, add 0.5 in the last row [should be np.nan, but not sure if theano could deal with that]

# Transform everything into theano.shared variables
prev_choices = theano.shared(np.asarray(prev_choices, dtype='int32'))
prev_choices_inv = theano.shared(np.asarray(prev_choices_inv, dtype='int32'))
rewards = theano.shared(np.asarray(rewards, dtype='int32'))
choices = theano.shared(np.asarray(choices, dtype='int32'))
# next_choices = theano.shared(np.asarray(next_choices, dtype='int32'))
group = theano.shared(np.asarray(group, dtype='int32'))

# Prepare things for saving
save_dir, save_id = get_save_dir_and_save_id(run_on_cluster, file_name_suff, fitted_data_name, n_samples)

# Fit models
model_dict = {}
print("Compiling models for {0} with {1} samples and {2} tuning steps...\n".format(fitted_data_name, n_samples, n_tune))

with pm.Model() as model:

    # # Get population-level and individual parameters
    # alpha_a_a = pm.Uniform('alpha_a_a', lower=0, upper=upper)
    # alpha_a_b = pm.Uniform('alpha_a_b', lower=0, upper=upper)
    # alpha_b_a = pm.Uniform('alpha_b_a', lower=0, upper=upper)
    # alpha_b_b = pm.Uniform('alpha_b_b', lower=0, upper=upper)
    alpha_a = pm.Gamma('alpha_a', alpha=1, beta=1, shape=n_groups)
    alpha_b = pm.Gamma('alpha_b', alpha=1, beta=1, shape=n_groups)

    # beta_a_a = pm.Uniform('beta_a_a', lower=0, upper=upper)
    # beta_a_b = pm.Uniform('beta_a_b', lower=0, upper=upper)
    # beta_b_a = pm.Uniform('beta_b_a', lower=0, upper=upper)
    # beta_b_b = pm.Uniform('beta_b_b', lower=0, upper=upper)
    beta_a = pm.Gamma('beta_a', alpha=1, beta=1, shape=n_groups)
    beta_b = pm.Gamma('beta_b', alpha=1, beta=1, shape=n_groups)

    # persev_mu_mu = pm.Uniform('persev_mu_mu', lower=-1, upper=1)
    # persev_mu_sd = pm.HalfNormal('persev_mu_sd', sd=0.5)
    # persev_sd_sd = pm.HalfNormal('persev_sd_sd', sd=0.1)

    # gamma_a = pm.Gamma('gamma_a', alpha=1, beta=1, shape=n_groups)
    # gamma_b = pm.Gamma('gamma_b', alpha=1, beta=1, shape=n_groups)

    # nalpha_a_a = pm.Uniform('nalpha_a_a', lower=0, upper=upper)
    # nalpha_a_b = pm.Uniform('nalpha_a_b', lower=0, upper=upper)
    # nalpha_b_a = pm.Uniform('nalpha_b_a', lower=0, upper=upper)
    # nalpha_b_b = pm.Uniform('nalpha_b_b', lower=0, upper=upper)
    nalpha_a = pm.Gamma('nalpha_a', alpha=1, beta=1, shape=n_groups)
    nalpha_b = pm.Gamma('nalpha_b', alpha=1, beta=1, shape=n_groups)

    # calpha_sc_a_a = pm.Uniform('calpha_sc_a_a', lower=0, upper=upper)
    # calpha_sc_a_b = pm.Uniform('calpha_sc_a_b', lower=0, upper=upper)
    # calpha_sc_b_a = pm.Uniform('calpha_sc_b_a', lower=0, upper=upper)
    # calpha_sc_b_b = pm.Uniform('calpha_sc_b_b', lower=0, upper=upper)
    calpha_sc_a = pm.Gamma('calpha_sc_a', alpha=1, beta=1, shape=n_groups)
    calpha_sc_b = pm.Gamma('calpha_sc_b', alpha=1, beta=1, shape=n_groups)
    calpha_sc = pm.Beta('calpha_sc', alpha=calpha_sc_a[group], beta=calpha_sc_b[group], shape=n_subj)
    # # calpha_sc = pm.Deterministic('calpha_sc', T.as_tensor_variable(1))
    #
    # # alpha_sc_a_a = pm.Uniform('alpha_sc_a_a', lower=0, upper=upper)
    # # alpha_sc_a_b = pm.Uniform('alpha_sc_a_b', lower=0, upper=upper)
    # # alpha_sc_b_a = pm.Uniform('alpha_sc_b_a', lower=0, upper=upper)
    # # alpha_sc_b_b = pm.Uniform('alpha_sc_b_b', lower=0, upper=upper)
    # # alpha_sc_a = pm.Gamma('alpha_sc_a', alpha=alpha_sc_a_a, beta=alpha_sc_a_b, shape=n_groups)
    # # alpha_sc_b = pm.Gamma('alpha_sc_b', alpha=alpha_sc_b_a, beta=alpha_sc_b_b, shape=n_groups)
    # # alpha_sc = pm.Beta('alpha_sc', alpha=alpha_sc_a[group], beta=alpha_sc_b[group], shape=n_subj)
    # # # alpha_sc = pm.Deterministic('alpha_sc', T.as_tensor_variable(1))
    #
    # # cnalpha_sc_a_a = pm.Uniform('cnalpha_sc_a_a', lower=0, upper=upper)
    # # cnalpha_sc_a_b = pm.Uniform('cnalpha_sc_a_b', lower=0, upper=upper)
    # # cnalpha_sc_b_a = pm.Uniform('cnalpha_sc_b_a', lower=0, upper=upper)
    # # cnalpha_sc_b_b = pm.Uniform('cnalpha_sc_b_b', lower=0, upper=upper)
    # # cnalpha_sc_a = pm.Gamma('cnalpha_sc_a', alpha=cnalpha_sc_a_a, beta=cnalpha_sc_a_b, shape=n_groups)
    # # cnalpha_sc_b = pm.Gamma('cnalpha_sc_b', alpha=cnalpha_sc_b_a, beta=cnalpha_sc_b_b, shape=n_groups)
    # # cnalpha_sc = pm.Beta('cnalpha_sc', alpha=cnalpha_sc_a[group], beta=cnalpha_sc_b[group], shape=n_subj)
    cnalpha_sc = pm.Deterministic('cnalpha_sc', calpha_sc.copy())

    # Get parameter means and variances
    alpha_mu = pm.Deterministic('alpha_mu', 1 / (1 + alpha_b / alpha_a))
    alpha_var = pm.Deterministic('alpha_var', (alpha_a * alpha_b) / (np.square(alpha_a + alpha_b) * (alpha_a + alpha_b + 1)))
    beta_mu = pm.Deterministic('beta_mu', beta_a / beta_b)
    beta_var = pm.Deterministic('beta_var', beta_a / np.square(beta_b))
    # persev_mu = pm.Bound(pm.Normal, lower=-1, upper=1)('persev_mu', mu=0, sd=0.1, shape=n_groups)
    # persev_sd = pm.HalfNormal('persev_sd', sd=0.1, shape=n_groups)
    nalpha_mu = pm.Deterministic('nalpha_mu', 1 / (1 + nalpha_b / nalpha_a))
    nalpha_var = pm.Deterministic('nalpha_var', (nalpha_a * nalpha_b) / (np.square(nalpha_a + nalpha_b) * (nalpha_a + nalpha_b + 1)))
    calpha_sc_mu = pm.Deterministic('calpha_sc_mu', 1 / (1 + calpha_sc_b / calpha_sc_a))
    calpha_sc_var = pm.Deterministic('calpha_sc_var', (calpha_sc_a * calpha_sc_b) / (np.square(calpha_sc_a + calpha_sc_b) * (calpha_sc_a + calpha_sc_b + 1)))
    # cnalpha_sc_mu = pm.Deterministic('cnalpha_sc_mu', 1 / (1 + cnalpha_sc_b / cnalpha_sc_a))
    # cnalpha_sc_var = pm.Deterministic('cnalpha_sc_var', (cnalpha_sc_a * cnalpha_sc_b) / (np.square(cnalpha_sc_a + cnalpha_sc_b) * (cnalpha_sc_a + cnalpha_sc_b + 1)))

    # Individual parameters
    alpha = pm.Beta('alpha', alpha=alpha_a[group], beta=alpha_b[group], shape=n_subj)
    beta = 1 + pm.Gamma('beta', alpha=beta_a[group], beta=beta_b[group], shape=n_subj)
    # persev = pm.Normal('persev', mu=persev_mu[group], sd=persev_sd[group], shape=(1, n_subj), testval=0.1 * T.ones((1, n_subj)))
    # gamma = pm.Beta('gamma', alpha=gamma_a[group], beta=gamma_b[group], shape=n_subj)
    # persev = T.zeros((1, n_subj))
    nalpha = pm.Beta('nalpha', alpha=nalpha_a[group], beta=nalpha_b[group], shape=n_subj)

    # alpha = pm.Beta('alpha', alpha=1, beta=1, shape=n_subj)
    # beta = 1 + pm.Gamma('beta', alpha=1, beta=1, shape=n_subj)
    # persev = pm.Normal('persev', mu=0, sd=0.1, shape=n_subj)
    # gamma = pm.Beta('gamma', alpha=1, beta=1, shape=n_subj)
    # nalpha = pm.Beta('nalpha', alpha=1, beta=1, shape=n_subj)
    # calpha_sc = pm.Beta('calpha_sc', alpha=5, beta=1, shape=n_subj)
    # cnalpha_sc = pm.Beta('cnalpha_sc', alpha=5, beta=1, shape=n_subj)
    eps = T.as_tensor_variable(0)

    calpha = pm.Deterministic('calpha', alpha * calpha_sc)
    cnalpha = pm.Deterministic('cnalpha', nalpha * cnalpha_sc)
    # persev_bonus_left = persev * prev_choices_inv
    # persev_bonus_right = persev * prev_choices
    # # Group differences?
    # alpha_mu_diff01 = pm.Deterministic('alpha_mu_diff01', alpha_mu[0] - alpha_mu[1])
    # alpha_mu_diff02 = pm.Deterministic('alpha_mu_diff02', alpha_mu[0] - alpha_mu[2])
    # alpha_mu_diff12 = pm.Deterministic('alpha_mu_diff12', alpha_mu[1] - alpha_mu[2])
    #
    # beta_mu_diff01 = pm.Deterministic('beta_mu_diff01', beta_mu[0] - beta_mu[1])
    # beta_mu_diff02 = pm.Deterministic('beta_mu_diff02', beta_mu[0] - beta_mu[2])
    # beta_mu_diff12 = pm.Deterministic('beta_mu_diff12', beta_mu[1] - beta_mu[2])
    #
    # persev_mu_diff01 = pm.Deterministic('persev_mu_diff01', persev_mu[0] - persev_mu[1])
    # persev_mu_diff02 = pm.Deterministic('persev_mu_diff02', persev_mu[0] - persev_mu[2])
    # persev_mu_diff12 = pm.Deterministic('persev_mu_diff12', persev_mu[1] - persev_mu[2])
    #
    # nalpha_mu_diff01 = pm.Deterministic('nalpha_mu_diff01', nalpha_mu[0] - nalpha_mu[1])
    # nalpha_mu_diff02 = pm.Deterministic('nalpha_mu_diff02', nalpha_mu[0] - nalpha_mu[2])
    # nalpha_mu_diff12 = pm.Deterministic('nalpha_mu_diff12', nalpha_mu[1] - nalpha_mu[2])
    #
    # calpha_sc_mu_diff01 = pm.Deterministic('calpha_sc_mu_diff01', calpha_sc_mu[0] - calpha_sc_mu[1])
    # calpha_sc_mu_diff02 = pm.Deterministic('calpha_sc_mu_diff02', calpha_sc_mu[0] - calpha_sc_mu[2])
    # calpha_sc_mu_diff12 = pm.Deterministic('calpha_sc_mu_diff12', calpha_sc_mu[1] - calpha_sc_mu[2])
    #
    # # cnalpha_sc_mu_diff01 = pm.Deterministic('cnalpha_sc_mu_diff01', cnalpha_sc_mu[0] - cnalpha_sc_mu[1])
    # # cnalpha_sc_mu_diff02 = pm.Deterministic('cnalpha_sc_mu_diff02', cnalpha_sc_mu[0] - cnalpha_sc_mu[2])
    # # cnalpha_sc_mu_diff12 = pm.Deterministic('cnalpha_sc_mu_diff12', cnalpha_sc_mu[1] - cnalpha_sc_mu[2])

    # Calculate Q-values
    # Q_left, Q_right = 0.5 * T.ones(n_subj, dtype='int32'), 0.5 * T.ones(n_subj, dtype='int32')
    Q_L1L_R1R, Q_L0L_R0R, Q_L1R_R1L, Q_L0R_R0L =\
        0.5 * T.ones(n_subj, dtype='int32'), 0.5 * T.ones(n_subj, dtype='int32'), 0.5 * T.ones(n_subj, dtype='int32'), 0.5 * T.ones(n_subj, dtype='int32')
    [Q_L1L_R1R, Q_L0L_R0R, Q_L1R_R1L, Q_L0R_R0L], _ = theano.scan(fn=update_Q,
                                       sequences=[choices[:-1], rewards[:-1], choices[1:], rewards[1:]],  # update after trial 2
                                       # outputs_info=[Q_left, Q_right],
                                       outputs_info=[Q_L1L_R1R, Q_L0L_R0R, Q_L1R_R1L, Q_L0R_R0L],
                                       non_sequences=[alpha, nalpha, calpha, cnalpha])  # gamma, alpha_sc])

    # Translate Q-values into probabilities and add eps noise
    p_right = p_from_Q(Q_L1L_R1R, Q_L0L_R0R, Q_L1R_R1L, Q_L0R_R0L,
                       choices[1:], rewards[1:],
                       # persev_bonus_left, persev_bonus_right,
                       beta, eps)

    # Add p=0.5 for the very first trial
    initial_p = 0.5 * T.ones((1, n_subj))
    p_right = T.concatenate([initial_p, p_right[:-1]], axis=0)
    theano.printing.Print('p_right')(p_right)

    # Use Bernoulli to sample responses
    model_choices = pm.Bernoulli('model_choices', p=p_right, observed=choices[1:])

    # Check model logp and RV logps (will crash if they are nan or -inf)
    if verbose or print_logps:
        print_logp_info(model)

    # Sample the model
    trace = pm.sample(n_samples, tune=n_tune, chains=n_chains, cores=n_cores, nuts_kwargs=dict(target_accept=target_accept))

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
