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
file_name_suff = 'abncnc'
upper = 1000

# Which data should be fitted?
fitted_data_name = 'humans'  # 'humans', 'simulations'
kids_and_teens_only = False
adults_only = False

# Sampling details
n_samples = 100
n_tune = 50
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
    # alpha_a_a = pm.Uniform('alpha_a_a', lower=0, upper=upper)
    # alpha_a_b = pm.Uniform('alpha_a_b', lower=0, upper=upper)
    # alpha_b_a = pm.Uniform('alpha_b_a', lower=0, upper=upper)
    # alpha_b_b = pm.Uniform('alpha_b_b', lower=0, upper=upper)
    # alpha_a = pm.Gamma('alpha_a', alpha=alpha_a_a, beta=alpha_a_b, shape=n_groups)
    # alpha_b = pm.Gamma('alpha_b', alpha=alpha_b_a, beta=alpha_b_b, shape=n_groups)

    beta_a_a = pm.Uniform('beta_a_a', lower=0, upper=upper)
    beta_a_b = pm.Uniform('beta_a_b', lower=0, upper=upper)
    beta_b_a = pm.Uniform('beta_b_a', lower=0, upper=upper)
    beta_b_b = pm.Uniform('beta_b_b', lower=0, upper=upper)
    beta_a = pm.Gamma('beta_a', alpha=beta_a_a, beta=beta_a_b, shape=n_groups)
    beta_b = pm.Gamma('beta_b', alpha=beta_b_a, beta=beta_b_b, shape=n_groups)
    eps = T.as_tensor_variable(0)

    nalpha_a_a = pm.Uniform('nalpha_a_a', lower=0, upper=upper)
    nalpha_a_b = pm.Uniform('nalpha_a_b', lower=0, upper=upper)
    nalpha_b_a = pm.Uniform('nalpha_b_a', lower=0, upper=upper)
    nalpha_b_b = pm.Uniform('nalpha_b_b', lower=0, upper=upper)
    nalpha_a = pm.Gamma('nalpha_a', alpha=nalpha_a_a, beta=nalpha_a_b, shape=n_groups)
    nalpha_b = pm.Gamma('nalpha_b', alpha=nalpha_b_a, beta=nalpha_b_b, shape=n_groups)

    calpha_sc_a_a = pm.Uniform('calpha_sc_a_a', lower=0, upper=upper)
    calpha_sc_a_b = pm.Uniform('calpha_sc_a_b', lower=0, upper=upper)
    calpha_sc_b_a = pm.Uniform('calpha_sc_b_a', lower=0, upper=upper)
    calpha_sc_b_b = pm.Uniform('calpha_sc_b_b', lower=0, upper=upper)
    calpha_sc_a = pm.Gamma('calpha_sc_a', alpha=calpha_sc_a_a, beta=calpha_sc_a_b, shape=n_groups)
    calpha_sc_b = pm.Gamma('calpha_sc_b', alpha=calpha_sc_b_a, beta=calpha_sc_b_b, shape=n_groups)
    calpha_sc = pm.Beta('calpha_sc', alpha=calpha_sc_a[group], beta=calpha_sc_b[group], shape=n_subj)
    # calpha_sc = pm.Deterministic('calpha_sc', T.as_tensor_variable(0))

    cnalpha_sc_a_a = pm.Uniform('cnalpha_sc_a_a', lower=0, upper=upper)
    cnalpha_sc_a_b = pm.Uniform('cnalpha_sc_a_b', lower=0, upper=upper)
    cnalpha_sc_b_a = pm.Uniform('cnalpha_sc_b_a', lower=0, upper=upper)
    cnalpha_sc_b_b = pm.Uniform('cnalpha_sc_b_b', lower=0, upper=upper)
    cnalpha_sc_a = pm.Gamma('cnalpha_sc_a', alpha=cnalpha_sc_a_a, beta=cnalpha_sc_a_b, shape=n_groups)
    cnalpha_sc_b = pm.Gamma('cnalpha_sc_b', alpha=cnalpha_sc_b_a, beta=cnalpha_sc_b_b, shape=n_groups)
    cnalpha_sc = pm.Beta('cnalpha_sc', alpha=cnalpha_sc_a[group], beta=cnalpha_sc_b[group], shape=n_subj)
    # cnalpha_sc = pm.Deterministic('cnalpha_sc', calpha_sc.copy())

    # Individual parameters
    alpha = pm.Deterministic('alpha', T.ones(n_subj))  # pm.Beta('alpha', alpha=alpha_a[group], beta=alpha_b[group], shape=n_subj)
    beta = pm.Gamma('beta', alpha=beta_a[group], beta=beta_b[group], shape=n_subj)
    nalpha = pm.Beta('nalpha', alpha=nalpha_a[group], beta=nalpha_b[group], shape=n_subj)
    nalpha = pm.Deterministic('nalpha', alpha.copy())
    calpha = pm.Deterministic('calpha', alpha * calpha_sc)
    cnalpha = pm.Deterministic('cnalpha', nalpha * cnalpha_sc)

    # Get parameter means and variances
    # alpha_mu = pm.Deterministic(
    #     'alpha_mu', 1 / (1 + alpha_b / alpha_a))
    # alpha_var = pm.Deterministic(
    #     'alpha_var', (alpha_a * alpha_b) / (np.square(alpha_a + alpha_b) * (alpha_a + alpha_b + 1)))
    beta_mu = pm.Deterministic(
        'beta_mu', beta_a / beta_b)
    beta_var = pm.Deterministic(
        'beta_var', beta_a / np.square(beta_b))
    nalpha_mu = pm.Deterministic(
        'nalpha_mu', 1 / (1 + nalpha_b / nalpha_a))
    nalpha_var = pm.Deterministic(
        'nalpha_var', (nalpha_a * nalpha_b) / (np.square(nalpha_a + nalpha_b) * (nalpha_a + nalpha_b + 1)))
    calpha_sc_mu = pm.Deterministic(
        'calpha_sc_mu', 1 / (1 + calpha_sc_b / calpha_sc_a))
    calpha_sc_var = pm.Deterministic(
        'calpha_sc_var', (calpha_sc_a * calpha_sc_b) / (np.square(calpha_sc_a + calpha_sc_b) * (calpha_sc_a + calpha_sc_b + 1)))
    cnalpha_sc_mu = pm.Deterministic(
        'cnalpha_sc_mu', 1 / (1 + cnalpha_sc_b / cnalpha_sc_a))
    cnalpha_sc_var = pm.Deterministic(
        'cnalpha_sc_var', (cnalpha_sc_a * cnalpha_sc_b) / (np.square(cnalpha_sc_a + cnalpha_sc_b) * (cnalpha_sc_a + cnalpha_sc_b + 1)))

    # Group differences?
    # alpha_mu_diff01 = pm.Deterministic('alpha_mu_diff01', alpha_a[0] - alpha_a[1])
    # alpha_mu_diff02 = pm.Deterministic('alpha_mu_diff02', alpha_a[0] - alpha_a[2])
    # alpha_mu_diff12 = pm.Deterministic('alpha_mu_diff12', alpha_a[1] - alpha_a[2])

    beta_mu_diff01 = pm.Deterministic('beta_mu_diff01', beta_a[0] - beta_a[1])
    beta_mu_diff02 = pm.Deterministic('beta_mu_diff02', beta_a[0] - beta_a[2])
    beta_mu_diff12 = pm.Deterministic('beta_mu_diff12', beta_a[1] - beta_a[2])

    nalpha_mu_diff01 = pm.Deterministic('nalpha_mu_diff01', nalpha_a[0] - nalpha_a[1])
    nalpha_mu_diff02 = pm.Deterministic('nalpha_mu_diff02', nalpha_a[0] - nalpha_a[2])
    nalpha_mu_diff12 = pm.Deterministic('nalpha_mu_diff12', nalpha_a[1] - nalpha_a[2])

    calpha_sc_mu_diff01 = pm.Deterministic('calpha_sc_mu_diff01', calpha_sc_a[0] - calpha_sc_a[1])
    calpha_sc_mu_diff02 = pm.Deterministic('calpha_sc_mu_diff02', calpha_sc_a[0] - calpha_sc_a[2])
    calpha_sc_mu_diff12 = pm.Deterministic('calpha_sc_mu_diff12', calpha_sc_a[1] - calpha_sc_a[2])

    cnalpha_sc_mu_diff01 = pm.Deterministic('cnalpha_sc_mu_diff01', cnalpha_sc_a[0] - cnalpha_sc_a[1])
    cnalpha_sc_mu_diff02 = pm.Deterministic('cnalpha_sc_mu_diff02', cnalpha_sc_a[0] - cnalpha_sc_a[2])
    cnalpha_sc_mu_diff12 = pm.Deterministic('cnalpha_sc_mu_diff12', cnalpha_sc_a[1] - cnalpha_sc_a[2])

    # Calculate Q-values
    Q_left, Q_right = 0.5 * T.ones(n_subj, dtype='int32'), 0.5 * T.ones(n_subj, dtype='int32')
    [Q_left, Q_right], _ = theano.scan(fn=update_Q,
                                       sequences=[rewards, choices],
                                       outputs_info=[Q_left, Q_right],
                                       non_sequences=[alpha, nalpha, calpha, cnalpha])

    # Translate Q-values into probabilities and add eps noise
    p_right = p_from_Q(Q_left, Q_right, beta, eps)

    # Add initial p=0.5 at the beginning of p_right
    initial_p = 0.5 * T.ones((1, n_subj))
    p_right = T.concatenate([initial_p, p_right[:-1]], axis=0)

    # Use Bernoulli to sample responses
    model_choices = pm.Bernoulli('model_choices', p=p_right, observed=choices)

    # Check model logp and RV logps (will crash if they are nan or -inf)
    if verbose or print_logps:
        print_logp_info(model)

    # Sample the model
    trace = pm.sample(n_samples, tune=n_tune, chains=n_chains, cores=n_cores, nuts_kwargs=dict(target_accept=target_accept))

# Get results
model_summary = pm.summary(trace)
model_summary.to_csv(save_dir + save_id + '_summary.csv')

if not run_on_cluster:
    pm.traceplot(trace)
    plt.savefig(save_dir + 'plot_' + save_id + '.png')

# Save results
print('Saving trace, trace plot, model, and model summary to {0}{1}...\n'.format(save_dir, save_id))
with open(save_dir + save_id + '.pickle', 'wb') as handle:
    pickle.dump({'trace': trace, 'model': model, 'summary': model_summary},
                handle, protocol=pickle.HIGHEST_PROTOCOL)
