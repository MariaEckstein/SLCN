import pickle

import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from shared_modeling_simulation import update_Q, p_from_Q, get_strat_Qs, get_WSLS_Qs
from modeling_helpers import load_data, get_save_dir_and_save_id, print_logp_info

import pymc3 as pm
import numpy as np

# Switches for this script
run_on_cluster = False
verbose = False
print_logps = False
file_name_suff = 'abcnSS'
upper = 1000
print("Working on model '{}'.".format(file_name_suff))

# Which data should be fitted?
fitted_data_name = 'humans'  # 'humans', 'simulations'
kids_and_teens_only = False
adults_only = False
n_groups = 3
n_subj = 'all'  # 'all' or int
n_trials = 'all'  # 'all' or int

# Sampling details
if not run_on_cluster:
    n_tune = 100
    n_samples = 100
    n_cores = 2
    n_chains = 1
else:
    n_tune = 2000
    n_samples = 5000
    n_cores = 10
    n_chains = 5
target_accept = 0.8

# Load to-be-fitted data from n_subj subjects with n_trials trials
n_subj, rewards, choices, group, n_groups, age_z = load_data(run_on_cluster, fitted_data_name, n_groups, kids_and_teens_only, adults_only, n_subj)
if n_trials == 'all':
    n_trials = len(choices)
else:
    choices = choices[:n_trials]
    rewards = rewards[:n_trials]

# Create choices_both (first trial is persev_bonus for second trial = where Qs starts)
choices_both = np.array([1 - choices, choices])  # left, right
choices_both = np.transpose(choices_both, (1, 2, 0))[:-1]  # make same shape as Qs (n_trials-1, n_subj, 2); remove last trial because it's not needed
# choices_both = np.concatenate([np.zeros((1, n_subj, 2)), choices_both[:-1]])  # add 0's for first trial (what I did previously, but should be wrong, and also doesn't converge)

# # Comment in for Strategy model ("stay unless failed to receive reward twice in a row for the same action.")
# Qs = get_strat_Qs(n_trials, n_subj)
# # Comment in for WSLS model ("stay if you won; switch if you lost")
# Qs = get_WSLS_Qs(n_trials, n_subj)
# Qs = theano.shared(np.asarray(Qs, dtype='float32'))

# Transform everything into theano.shared variables
rewards = theano.shared(np.asarray(rewards, dtype='int32'))
choices = theano.shared(np.asarray(choices, dtype='int32'))
choices_both = theano.shared(np.asarray(choices_both, dtype='int32'))
group = theano.shared(np.asarray(group, dtype='int32'))

# Prepare things for saving
save_dir, save_id = get_save_dir_and_save_id(run_on_cluster, file_name_suff, fitted_data_name, n_samples)

# Fit models
model_dict = {}
print("Compiling models for {0} with {1} samples and {2} tuning steps...\n".format(fitted_data_name, n_samples, n_tune))

with pm.Model() as model:

    # Get population-level and individual parameters
    alpha_a_a = pm.Uniform('alpha_a_a', lower=0, upper=upper)
    alpha_a_b = pm.Uniform('alpha_a_b', lower=0, upper=upper)
    alpha_b_a = pm.Uniform('alpha_b_a', lower=0, upper=upper)
    alpha_b_b = pm.Uniform('alpha_b_b', lower=0, upper=upper)
    alpha_a = pm.Gamma('alpha_a', alpha=alpha_a_a, beta=alpha_a_b, shape=n_groups)
    alpha_b = pm.Gamma('alpha_b', alpha=alpha_b_a, beta=alpha_b_b, shape=n_groups)

    beta_a_a = pm.Uniform('beta_a_a', lower=0, upper=upper)
    beta_a_b = pm.Uniform('beta_a_b', lower=0, upper=upper)
    beta_b_a = pm.Uniform('beta_b_a', lower=0, upper=upper)
    beta_b_b = pm.Uniform('beta_b_b', lower=0, upper=upper)
    beta_a = pm.Gamma('beta_a', alpha=beta_a_a, beta=beta_a_b, shape=n_groups)
    beta_b = pm.Gamma('beta_b', alpha=beta_b_a, beta=beta_b_b, shape=n_groups)

    # persev_mu_mu = pm.Uniform('persev_mu_mu', lower=-1, upper=1)
    # persev_mu_sd = pm.HalfNormal('persev_mu_sd', sd=0.5)
    # persev_sd_sd = pm.HalfNormal('persev_sd_sd', sd=0.1)

    # gamma_a = pm.Gamma('gamma_a', alpha=1, beta=1, shape=n_groups)
    # gamma_b = pm.Gamma('gamma_b', alpha=1, beta=1, shape=n_groups)

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
    calpha_sc = pm.Beta('calpha_sc', alpha=calpha_sc_a[group], beta=calpha_sc_b[group], shape=n_subj, testval=0.25 * T.ones(n_subj))
    # calpha_sc = pm.Deterministic('calpha_sc', T.as_tensor_variable(1))

    # alpha_sc_a_a = pm.Uniform('alpha_sc_a_a', lower=0, upper=upper)
    # alpha_sc_a_b = pm.Uniform('alpha_sc_a_b', lower=0, upper=upper)
    # alpha_sc_b_a = pm.Uniform('alpha_sc_b_a', lower=0, upper=upper)
    # alpha_sc_b_b = pm.Uniform('alpha_sc_b_b', lower=0, upper=upper)
    # alpha_sc_a = pm.Gamma('alpha_sc_a', alpha=alpha_sc_a_a, beta=alpha_sc_a_b, shape=n_groups)
    # alpha_sc_b = pm.Gamma('alpha_sc_b', alpha=alpha_sc_b_a, beta=alpha_sc_b_b, shape=n_groups)
    # alpha_sc = pm.Beta('alpha_sc', alpha=alpha_sc_a[group], beta=alpha_sc_b[group], shape=n_subj)
    # # alpha_sc = pm.Deterministic('alpha_sc', T.as_tensor_variable(1))

    # # cnalpha_sc_a_a = pm.Uniform('cnalpha_sc_a_a', lower=0, upper=upper)
    # # cnalpha_sc_a_b = pm.Uniform('cnalpha_sc_a_b', lower=0, upper=upper)
    # # cnalpha_sc_b_a = pm.Uniform('cnalpha_sc_b_a', lower=0, upper=upper)
    # # cnalpha_sc_b_b = pm.Uniform('cnalpha_sc_b_b', lower=0, upper=upper)
    # # cnalpha_sc_a = pm.Gamma('cnalpha_sc_a', alpha=cnalpha_sc_a_a, beta=cnalpha_sc_a_b, shape=n_groups)
    # # cnalpha_sc_b = pm.Gamma('cnalpha_sc_b', alpha=cnalpha_sc_b_a, beta=cnalpha_sc_b_b, shape=n_groups)
    # # cnalpha_sc = pm.Beta('cnalpha_sc', alpha=cnalpha_sc_a[group], beta=cnalpha_sc_b[group], shape=n_subj)
    cnalpha_sc = pm.Deterministic('cnalpha_sc', calpha_sc.copy())
    #
    # # Get parameter means and variances
    alpha_mu = pm.Deterministic('alpha_mu', 1 / (1 + alpha_b / alpha_a))
    alpha_var = pm.Deterministic('alpha_var', (alpha_a * alpha_b) / (np.square(alpha_a + alpha_b) * (alpha_a + alpha_b + 1)))
    beta_mu = pm.Deterministic('beta_mu', beta_a / beta_b + 1)
    beta_var = pm.Deterministic('beta_var', beta_a / np.square(beta_b))
    # # persev_mu = pm.Bound(pm.Normal, lower=-1, upper=1)('persev_mu', mu=persev_mu_mu, sd=persev_mu_sd, shape=n_groups)
    # # persev_sd = pm.HalfNormal('persev_sd', sd=persev_sd_sd, shape=n_groups)
    nalpha_mu = pm.Deterministic('nalpha_mu', 1 / (1 + nalpha_b / nalpha_a))
    nalpha_var = pm.Deterministic('nalpha_var', (nalpha_a * nalpha_b) / (np.square(nalpha_a + nalpha_b) * (nalpha_a + nalpha_b + 1)))
    calpha_sc_mu = pm.Deterministic('calpha_sc_mu', 1 / (1 + calpha_sc_b / calpha_sc_a))
    calpha_sc_var = pm.Deterministic('calpha_sc_var', (calpha_sc_a * calpha_sc_b) / (np.square(calpha_sc_a + calpha_sc_b) * (calpha_sc_a + calpha_sc_b + 1)))
    # # cnalpha_sc_mu = pm.Deterministic('cnalpha_sc_mu', 1 / (1 + cnalpha_sc_b / cnalpha_sc_a))
    # # cnalpha_sc_var = pm.Deterministic('cnalpha_sc_var', (cnalpha_sc_a * cnalpha_sc_b) / (np.square(cnalpha_sc_a + cnalpha_sc_b) * (cnalpha_sc_a + cnalpha_sc_b + 1)))
    #
    # # Individual parameters
    alpha = pm.Beta('alpha', alpha=alpha_a[group], beta=alpha_b[group], shape=n_subj, testval=0.5 * T.ones(n_subj))
    b = pm.Gamma('b', alpha=beta_a[group], beta=beta_b[group], shape=n_subj, testval=T.ones(n_subj))
    beta = pm.Deterministic('beta', b + 1)
    persev = 0  # (comment in for S, SS, and strat models)
    # # persev = pm.Normal('persev', mu=persev_mu[group], sd=persev_sd[group],
    # #                    shape=(1, n_subj, 1), testval=0.1 * T.ones((1, n_subj, 1)))  # comment in for letter model
    persev_bonus = persev * choices_both
    #
    # # gamma = pm.Beta('gamma', alpha=gamma_a[group], beta=gamma_b[group], shape=n_subj)
    nalpha = pm.Beta('nalpha', alpha=nalpha_a[group], beta=nalpha_b[group], shape=n_subj, testval=0.5 * T.ones(n_subj))
    calpha = pm.Deterministic('calpha', alpha * calpha_sc)
    cnalpha = pm.Deterministic('cnalpha', nalpha * cnalpha_sc)

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

    # Initialize Q-values for a single trial
    # # Comment in for letter model (using choice, reward)
    # Qs, _ = 0.5 * T.ones((n_subj, 2), dtype='int32'), 0.5 * T.ones(n_subj, dtype='int32')
    # # Comment in for S model (using prev_choice, prev_reward, choice, reward)
    # Qs = 0.5 * T.ones((n_subj, 2, 2, 2), dtype='int32')  # shape: (n_subj, prev_choice, prev_reward, choice)
    # # Comment in for SS model (using prev_prev_choice, prev_prev_reward, prev_choice, prev_reward, choice, reward)
    Qs = 0.5 * T.ones((n_subj, 2, 2, 2, 2, 2), dtype='float32')  # shape: (n_subj, prev_prev_choice, prev_prev_reward, prev_choice, prev_reward, choice)
    _ = T.ones(n_subj, dtype='float32')

    # Calculate Q-values for all trials
    [Qs, _], _ = theano.scan(  # shape: (n_trials-2, n_subj, prev_choice, prev_reward, choice)
        fn=update_Q,
        sequences=[
            choices[0:-2], rewards[0:-2],  # prev_prev_choice, prev_prev_reward: 1st trial til 3rd-to-last
            choices[1:-1], rewards[1:-1],  # prev_choice, prev_reward: 2nd trial til 2nd-to-last
            choices[2:], rewards[2:]],  # choice, reward: 3rd trial til last (for updating)
        outputs_info=[Qs, _],
        non_sequences=[alpha, nalpha, calpha, cnalpha, n_subj])

    # Initialize p_right for a single trial
    p_right = 0.5 * T.ones(n_subj, dtype='float32')  # shape: (n_subj)

    # Translate Q-values into probabilities for all trials
    p_right, _ = theano.scan(  # shape: (n_trials-2, n_subj)
        fn=p_from_Q,
        sequences=[Qs, persev_bonus[1:],
                   choices[0:-2], rewards[0:-2],
                   choices[1:-1], rewards[1:-1]
                   ],
        outputs_info=[p_right],
        non_sequences=[n_subj, beta])

    # Use Bernoulli to sample responses
    model_choices = pm.Bernoulli('model_choices', p=p_right, observed=choices[2:])  # predict from 3rd trial on

    # Check model logp and RV logps (will crash if they are nan or -inf)
    if verbose or print_logps:
        print_logp_info(model)
        theano.printing.Print('choices')(choices)
        theano.printing.Print('rewards')(rewards)
        theano.printing.Print('Qs')(Qs)
        # theano.printing.Print('Qs_p')(Qs + persev_bonus[1:])  # (comment in when using persev_bonus)
        theano.printing.Print('p_right')(p_right)

    # Sample the model
    trace = pm.sample(n_samples, tune=n_tune, chains=n_chains, cores=n_cores, nuts_kwargs=dict(target_accept=target_accept))
    # map = pm.find_MAP()

# Get results
# print("MAP estimate: {}".format(map))
model_summary = pm.summary(trace)
waic = pm.waic(trace, model)

if not run_on_cluster:
    pm.traceplot(trace)
    plt.savefig("{0}plot_{1}_{2}.png".format(save_dir, save_id, np.round(waic.WAIC, 2)))

# Save results
print('Saving trace, trace plot, model, and model summary to {0}{1}...\n'.format(save_dir, save_id))
with open(save_dir + save_id + '.pickle', 'wb') as handle:
    pickle.dump({'trace': trace, 'model': model, 'summary': model_summary, 'WAIC': waic.WAIC},
                handle, protocol=pickle.HIGHEST_PROTOCOL)
