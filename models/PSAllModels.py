import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import pymc3 as pm
import seaborn as sns
import theano
import theano.tensor as T
# floatX = 'float32'
# theano.config.floatX = 'float32'
theano.config.warn_float64 = 'warn'

from shared_modeling_simulation import *  # get_WSLSS_Qs, get_WSLS_Qs, update_Q_0, update_Q_1, update_Q_2, p_from_Q_0, p_from_Q_1, p_from_Q_2, get_likelihoods, post_from_lik, get_n_trials_back
from modeling_helpers import load_data, get_save_dir_and_save_id, print_logp_info


run_on_cluster = False
fit_mcmc = False
fit_map = True
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


def create_model(choices, rewards, group,
                 n_subj='all', n_trials='all',  # 'all' or int
                 model_name='ab',  # ab, abc, abn, abcn, abcnx, abcnS, abcnSS, WSLS, WSLSS, etc. etc.
                 verbose=False, print_logps=False,
                 fitted_data_name='humans',  # 'humans', 'simulations'
                 n_groups=3, fit_individuals=True,
                 upper=1000,
                 ):

    # Debug: smaller data set
    if n_trials == 'all':
        n_trials = len(choices)
    else:
        choices = choices[:n_trials]
        rewards = rewards[:n_trials]

    # Create choices_both (first trial is persev_bonus for second trial = where Qs starts)
    if 'B' in model_name:
        persev_bonus = 2 * choices - 1  # recode as -1 for choice==0 (left) and +1 for choice==1 (right)
        persev_bonus = np.concatenate([np.zeros((1, n_subj)), persev_bonus])  # add 0 bonus for first trial
        persev_bonus = theano.shared(np.asarray(persev_bonus, dtype='int16'))

    # Transform everything into theano.shared variables
    rewards = theano.shared(np.asarray(rewards, dtype='int16'))
    choices = theano.shared(np.asarray(choices, dtype='int16'))
    group = theano.shared(np.asarray(group, dtype='int16'))

    # Get n_trials_back, n_params, and file_name_suff
    n_trials_back = get_n_trials_back(model_name)
    n_params = get_n_params(model_name)
    file_name_suff = model_name + ''
    save_dir, save_id = get_save_dir_and_save_id(run_on_cluster, file_name_suff, fitted_data_name, n_samples)
    print("Working on model '{0}', which has {1} free parameters.".format(model_name, n_params))

    # Get fixed Q-values for WSLS and WSLSS
    if 'WSLSS' in model_name:  # "stay unless fail to receive reward twice in a row for the same action."
        Qs = get_WSLSS_Qs(n_trials, n_subj)
        Qs = theano.shared(np.asarray(Qs, dtype='float32'))

    elif 'WSLS' in model_name:  # "stay if you won; switch if you lost"
        Qs = get_WSLS_Qs(n_trials, n_subj)
        Qs = theano.shared(np.asarray(Qs, dtype='float32'))

    print("Compiling models for {0} with {1} samples and {2} tuning steps...\n".format(fitted_data_name, n_samples, n_tune))
    with pm.Model() as model:
        if not fit_individuals:
            a = 3
            # if 'RL' in model_name:
            #     # Get population-level and individual parameters
            #     alpha_a_a = pm.Uniform('alpha_a_a', lower=0, upper=upper)
            #     alpha_a_b = pm.Uniform('alpha_a_b', lower=0, upper=upper)
            #     alpha_b_a = pm.Uniform('alpha_b_a', lower=0, upper=upper)
            #     alpha_b_b = pm.Uniform('alpha_b_b', lower=0, upper=upper)
            #     alpha_a = pm.Gamma('alpha_a', alpha=alpha_a_a, beta=alpha_a_b, shape=n_groups)
            #     alpha_b = pm.Gamma('alpha_b', alpha=alpha_b_a, beta=alpha_b_b, shape=n_groups)
            #
            #     beta_a_a = pm.Uniform('beta_a_a', lower=0, upper=upper)
            #     beta_a_b = pm.Uniform('beta_a_b', lower=0, upper=upper)
            #     beta_b_a = pm.Uniform('beta_b_a', lower=0, upper=upper)
            #     beta_b_b = pm.Uniform('beta_b_b', lower=0, upper=upper)
            #     beta_a = pm.Gamma('beta_a', alpha=beta_a_a, beta=beta_a_b, shape=n_groups)
            #     beta_b = pm.Gamma('beta_b', alpha=beta_b_a, beta=beta_b_b, shape=n_groups)
            #
            #     nalpha_a_a = pm.Uniform('nalpha_a_a', lower=0, upper=upper)
            #     nalpha_a_b = pm.Uniform('nalpha_a_b', lower=0, upper=upper)
            #     nalpha_b_a = pm.Uniform('nalpha_b_a', lower=0, upper=upper)
            #     nalpha_b_b = pm.Uniform('nalpha_b_b', lower=0, upper=upper)
            #     nalpha_a = pm.Gamma('nalpha_a', alpha=nalpha_a_a, beta=nalpha_a_b, shape=n_groups)
            #     nalpha_b = pm.Gamma('nalpha_b', alpha=nalpha_b_a, beta=nalpha_b_b, shape=n_groups)
            #
            #     calpha_sc_a_a = pm.Uniform('calpha_sc_a_a', lower=0, upper=upper)
            #     calpha_sc_a_b = pm.Uniform('calpha_sc_a_b', lower=0, upper=upper)
            #     calpha_sc_b_a = pm.Uniform('calpha_sc_b_a', lower=0, upper=upper)
            #     calpha_sc_b_b = pm.Uniform('calpha_sc_b_b', lower=0, upper=upper)
            #     calpha_sc_a = pm.Gamma('calpha_sc_a', alpha=calpha_sc_a_a, beta=calpha_sc_a_b, shape=n_groups)
            #     calpha_sc_b = pm.Gamma('calpha_sc_b', alpha=calpha_sc_b_a, beta=calpha_sc_b_b, shape=n_groups)
            #     calpha_sc = pm.Beta('calpha_sc', alpha=calpha_sc_a[group], beta=calpha_sc_b[group], shape=n_subj, testval=0.25 * T.ones(n_subj))
            #     # calpha_sc = pm.Deterministic('calpha_sc', T.as_tensor_variable(1))
            #
            #     # cnalpha_sc_a_a = pm.Uniform('cnalpha_sc_a_a', lower=0, upper=upper)
            #     # cnalpha_sc_a_b = pm.Uniform('cnalpha_sc_a_b', lower=0, upper=upper)
            #     # cnalpha_sc_b_a = pm.Uniform('cnalpha_sc_b_a', lower=0, upper=upper)
            #     # cnalpha_sc_b_b = pm.Uniform('cnalpha_sc_b_b', lower=0, upper=upper)
            #     # cnalpha_sc_a = pm.Gamma('cnalpha_sc_a', alpha=cnalpha_sc_a_a, beta=cnalpha_sc_a_b, shape=n_groups)
            #     # cnalpha_sc_b = pm.Gamma('cnalpha_sc_b', alpha=cnalpha_sc_b_a, beta=cnalpha_sc_b_b, shape=n_groups)
            #     # cnalpha_sc = pm.Beta('cnalpha_sc', alpha=cnalpha_sc_a[group], beta=cnalpha_sc_b[group], shape=n_subj)
            #     cnalpha_sc = pm.Deterministic('cnalpha_sc', calpha_sc.copy())
            #
            #     # Get parameter means and variances
            #     alpha_mu = pm.Deterministic('alpha_mu', 1 / (1 + alpha_b / alpha_a))
            #     alpha_var = pm.Deterministic('alpha_var', (alpha_a * alpha_b) / (np.square(alpha_a + alpha_b) * (alpha_a + alpha_b + 1)))
            #     beta_mu = pm.Deterministic('beta_mu', beta_a / beta_b)
            #     beta_var = pm.Deterministic('beta_var', beta_a / np.square(beta_b))
            #     nalpha_mu = pm.Deterministic('nalpha_mu', 1 / (1 + nalpha_b / nalpha_a))
            #     nalpha_var = pm.Deterministic('nalpha_var', (nalpha_a * nalpha_b) / (np.square(nalpha_a + nalpha_b) * (nalpha_a + nalpha_b + 1)))
            #     calpha_sc_mu = pm.Deterministic('calpha_sc_mu', 1 / (1 + calpha_sc_b / calpha_sc_a))
            #     calpha_sc_var = pm.Deterministic('calpha_sc_var', (calpha_sc_a * calpha_sc_b) / (np.square(calpha_sc_a + calpha_sc_b) * (calpha_sc_a + calpha_sc_b + 1)))
            #     # cnalpha_sc_mu = pm.Deterministic('cnalpha_sc_mu', 1 / (1 + cnalpha_sc_b / cnalpha_sc_a))
            #     # cnalpha_sc_var = pm.Deterministic('cnalpha_sc_var', (cnalpha_sc_a * cnalpha_sc_b) / (np.square(cnalpha_sc_a + cnalpha_sc_b) * (cnalpha_sc_a + cnalpha_sc_b + 1)))
            #
            #     # Individual parameters (with group-level priors)
            #     alpha = pm.Beta('alpha', alpha=alpha_a[group], beta=alpha_b[group], shape=n_subj, testval=0.5 * T.ones(n_subj))
            #     beta = pm.Gamma('beta', alpha=beta_a[group], beta=beta_b[group], shape=n_subj, testval=T.ones(n_subj))
            #     persev = 0
            #     persev_bonus = persev * choices_both
            #
            #     nalpha = pm.Beta('nalpha', alpha=nalpha_a[group], beta=nalpha_b[group], shape=n_subj, testval=0.5 * T.ones(n_subj))
            #     calpha = pm.Deterministic('calpha', alpha * calpha_sc)
            #     cnalpha = pm.Deterministic('cnalpha', nalpha * cnalpha_sc)
            #
            #     # Group differences?
            #     alpha_mu_diff01 = pm.Deterministic('alpha_mu_diff01', alpha_mu[0] - alpha_mu[1])
            #     alpha_mu_diff02 = pm.Deterministic('alpha_mu_diff02', alpha_mu[0] - alpha_mu[2])
            #     alpha_mu_diff12 = pm.Deterministic('alpha_mu_diff12', alpha_mu[1] - alpha_mu[2])
            #
            #     beta_mu_diff01 = pm.Deterministic('beta_mu_diff01', beta_mu[0] - beta_mu[1])
            #     beta_mu_diff02 = pm.Deterministic('beta_mu_diff02', beta_mu[0] - beta_mu[2])
            #     beta_mu_diff12 = pm.Deterministic('beta_mu_diff12', beta_mu[1] - beta_mu[2])
            #
            #     nalpha_mu_diff01 = pm.Deterministic('nalpha_mu_diff01', nalpha_mu[0] - nalpha_mu[1])
            #     nalpha_mu_diff02 = pm.Deterministic('nalpha_mu_diff02', nalpha_mu[0] - nalpha_mu[2])
            #     nalpha_mu_diff12 = pm.Deterministic('nalpha_mu_diff12', nalpha_mu[1] - nalpha_mu[2])
            #
            #     calpha_sc_mu_diff01 = pm.Deterministic('calpha_sc_mu_diff01', calpha_sc_mu[0] - calpha_sc_mu[1])
            #     calpha_sc_mu_diff02 = pm.Deterministic('calpha_sc_mu_diff02', calpha_sc_mu[0] - calpha_sc_mu[2])
            #     calpha_sc_mu_diff12 = pm.Deterministic('calpha_sc_mu_diff12', calpha_sc_mu[1] - calpha_sc_mu[2])
            #
            #     # cnalpha_sc_mu_diff01 = pm.Deterministic('cnalpha_sc_mu_diff01', cnalpha_sc_mu[0] - cnalpha_sc_mu[1])
            #     # cnalpha_sc_mu_diff02 = pm.Deterministic('cnalpha_sc_mu_diff02', cnalpha_sc_mu[0] - cnalpha_sc_mu[2])
            #     # cnalpha_sc_mu_diff12 = pm.Deterministic('cnalpha_sc_mu_diff12', cnalpha_sc_mu[1] - cnalpha_sc_mu[2])
            #
            # elif 'B' in model_name:
            #
            #     # Get population-level and individual parameters
            #     p_noisy = 1e-5 * T.as_tensor_variable(1)
            #     beta_a_a = pm.Uniform('beta_a_a', lower=0, upper=upper)
            #     beta_a_b = pm.Uniform('beta_a_b', lower=0, upper=upper)
            #     beta_b_a = pm.Uniform('beta_b_a', lower=0, upper=upper)
            #     beta_b_b = pm.Uniform('beta_b_b', lower=0, upper=upper)
            #     beta_a = pm.Gamma('beta_a', alpha=beta_a_a, beta=beta_a_b, shape=n_groups)
            #     beta_b = pm.Gamma('beta_b', alpha=beta_b_a, beta=beta_b_b, shape=n_groups)
            #
            #     persev_mu_mu = pm.Uniform('persev_mu_mu', lower=-1, upper=1)
            #     persev_mu_sd = pm.HalfNormal('persev_mu_sd', sd=0.5)
            #     persev_sd_sd = pm.HalfNormal('persev_sd_sd', sd=0.1)
            #
            #     p_switch_a_a = pm.Uniform('p_switch_a_a', lower=0, upper=upper)
            #     p_switch_a_b = pm.Uniform('p_switch_a_b', lower=0, upper=upper)
            #     p_switch_b_a = pm.Uniform('p_switch_b_a', lower=0, upper=upper)
            #     p_switch_b_b = pm.Uniform('p_switch_b_b', lower=0, upper=upper)
            #     p_switch_a = pm.Gamma('p_switch_a', alpha=p_switch_a_a, beta=p_switch_a_b, shape=n_groups)
            #     p_switch_b = pm.Gamma('p_switch_b', alpha=p_switch_b_a, beta=p_switch_b_b, shape=n_groups)
            #
            #     p_reward_a_a = pm.Uniform('p_reward_a_a', lower=0, upper=upper)
            #     p_reward_a_b = pm.Uniform('p_reward_a_b', lower=0, upper=upper)
            #     p_reward_b_a = pm.Uniform('p_reward_b_a', lower=0, upper=upper)
            #     p_reward_b_b = pm.Uniform('p_reward_b_b', lower=0, upper=upper)
            #     p_reward_a = pm.Gamma('p_reward_a', alpha=p_reward_a_a, beta=p_reward_a_b, shape=n_groups)
            #     p_reward_b = pm.Gamma('p_reward_b', alpha=p_reward_b_a, beta=p_reward_b_b, shape=n_groups)
            #
            #     # Parameter mu and var and group differences
            #     beta_mu = pm.Deterministic('beta_mu', beta_a / beta_b)
            #     beta_var = pm.Deterministic('beta_var', beta_a / T.square(beta_b))
            #     persev_mu = pm.Bound(pm.Normal, lower=-1, upper=1)('persev_mu', mu=persev_mu_mu, sd=persev_mu_sd,
            #                                                        shape=n_groups)
            #     persev_sd = pm.HalfNormal('persev_sd', sd=persev_sd_sd, shape=n_groups)
            #     p_switch_mu = pm.Deterministic('p_switch_mu', p_switch_a / (p_switch_a + p_switch_b))
            #     p_switch_var = pm.Deterministic('p_switch_var', p_switch_a * p_switch_b / (
            #                 T.square(p_switch_a + p_switch_b) * (p_switch_a + p_switch_b + 1)))
            #     p_reward_mu = pm.Deterministic('p_reward_mu', p_reward_a / (p_reward_a + p_reward_b))
            #     p_reward_var = pm.Deterministic('p_reward_var', p_reward_a * p_reward_b / (
            #                 T.square(p_reward_a + p_reward_b) * (p_reward_a + p_reward_b + 1)))
            #
            #     # Individual parameters
            #     beta = pm.Gamma('beta', alpha=beta_a[group], beta=beta_b[group], shape=n_subj)
            #     # beta = T.ones(n_subj)  # TODO: COMMENT OUT `p_choice = 1 / (1 + np.exp(-beta * (p_choice - (1 - p_choice))))` FOR MODEL WITHOUT BETA (MODEL WITHOUT BETA CAN'T HAVE PERSEVERANCE)!
            #     persev = pm.Bound(pm.Normal, lower=-1, upper=1)('persev', mu=persev_mu[group], sd=persev_sd[group],
            #                                                     shape=(1, n_subj), testval=0.1 * T.ones((1, n_subj)))
            #     scaled_persev_bonus = persev_bonus * persev
            #     p_switch = pm.Beta('p_switch', alpha=p_switch_a[group], beta=p_switch_b[group], shape=n_subj)
            #     p_reward = pm.Beta('p_reward', alpha=p_reward_a[group], beta=p_reward_b[group], shape=n_subj)
            #
            #     # Group differences
            #     beta_mu_diff01 = pm.Deterministic('beta_mu_diff01', beta_mu[0] - beta_mu[1])
            #     beta_mu_diff02 = pm.Deterministic('beta_mu_diff02', beta_mu[0] - beta_mu[2])
            #     beta_mu_diff12 = pm.Deterministic('beta_mu_diff12', beta_mu[1] - beta_mu[2])
            #     persev_mu_diff01 = pm.Deterministic('persev_mu_diff01', persev_mu[0] - persev_mu[1])
            #     persev_mu_diff02 = pm.Deterministic('persev_mu_diff02', persev_mu[0] - persev_mu[2])
            #     persev_mu_diff12 = pm.Deterministic('persev_mu_diff12', persev_mu[1] - persev_mu[2])
            #     p_switch_mu_diff01 = pm.Deterministic('p_switch_mu_diff01', p_switch_mu[0] - p_switch_mu[1])
            #     p_switch_mu_diff02 = pm.Deterministic('p_switch_mu_diff02', p_switch_mu[0] - p_switch_mu[2])
            #     p_switch_mu_diff12 = pm.Deterministic('p_switch_mu_diff12', p_switch_mu[1] - p_switch_mu[2])
            #     p_reward_mu_diff01 = pm.Deterministic('p_reward_mu_diff01', p_reward_mu[0] - p_reward_mu[1])
            #     p_reward_mu_diff02 = pm.Deterministic('p_reward_mu_diff02', p_reward_mu[0] - p_reward_mu[2])
            #     p_reward_mu_diff12 = pm.Deterministic('p_reward_mu_diff12', p_reward_mu[1] - p_reward_mu[2])

        else:  # if fit_individuals == True:

            beta = pm.Gamma('beta', alpha=1, beta=1, shape=n_subj)
            if verbose:
                print("Adding free parameter beta.")
            if 'p' in model_name:
                persev = pm.Bound(pm.Normal, lower=-1, upper=1)(
                    'persev', mu=0, sd=0.2, shape=(n_subj), testval=0.1 * T.ones(n_subj, dtype='float32'))
                if verbose:
                    print("Adding free parameter persev.")
            else:
                persev = pm.Deterministic('persev', T.zeros(n_subj, dtype='float32'))

            if 'RL' in model_name:
                if 'a' in model_name:
                    alpha = pm.Beta('alpha', alpha=1, beta=1, shape=n_subj, testval=0.5 * T.ones(n_subj, dtype='float32'))
                    if verbose:
                        print("Adding free parameter alpha.")
                else:
                    alpha = pm.Deterministic('alpha', T.ones(n_subj, dtype='float32'))
                if 'n' in model_name:
                    nalpha = pm.Beta('nalpha', alpha=1, beta=1, shape=n_subj, testval=0.25 * T.ones(n_subj, dtype='float32'))
                    if verbose:
                        print("Adding free parameter nalpha.")
                else:
                    nalpha = pm.Deterministic('nalpha', 1 * alpha)
                    if verbose:
                        print("Setting nalpha = alpha.")
                if 'c' in model_name:
                    calpha_sc = pm.Beta('calpha_sc', alpha=1, beta=1, shape=n_subj, testval=0.25 * T.ones(n_subj, dtype='float32'))
                    if verbose:
                        print("Adding free parameter calpha.")
                else:
                    calpha_sc = 0
                if 'x' in model_name:
                    cnalpha_sc = pm.Beta('cnalpha_sc', alpha=1, beta=1, shape=n_subj, testval=0.25 * T.ones(n_subj, dtype='float32'))
                    if verbose:
                        print("Adding free parameter cnalpha.")
                elif 'c' in model_name:
                    cnalpha_sc = pm.Deterministic('cnalpha_sc', 1 * calpha_sc)
                    if verbose:
                        print("Setting cnalpha_sc = calpha_sc.")
                else:
                    cnalpha_sc = 0
                calpha = pm.Deterministic('calpha', alpha * calpha_sc)
                cnalpha = pm.Deterministic('cnalpha', nalpha * cnalpha_sc)

            elif 'B' in model_name:
                scaled_persev_bonus = persev_bonus * persev
                p_noisy = pm.Deterministic('p_noisy', 1e-5 * T.ones(n_subj, dtype='float32'))
                if 's' in model_name:
                    p_switch = pm.Beta('p_switch', alpha=1, beta=1, shape=n_subj, testval=0.1 * T.ones(n_subj))
                    if verbose:
                        print("Adding free parameter p_switch.")
                else:
                    p_switch = pm.Deterministic('p_switch', 0.05081582 * T.ones(n_subj))  # checked on 2019-06-03 in R: `mean(all_files$switch_trial)`
                if 'r' in model_name:
                    p_reward = pm.Beta('p_reward', alpha=1, beta=1, shape=n_subj, testval=0.75 * T.ones(n_subj))
                    if verbose:
                        print("Adding free p_reward.")
                else:
                    p_reward = pm.Deterministic('p_reward', 0.75 * T.ones(n_subj))  # 0.75 because p_reward is the prob. of getting reward if choice is correct

        # Initialize Q-values (with and without bias, i.e., option 'i')
        if 'RL' in model_name:

            if 'SS' in model_name:  # SS models
                if 'i' in model_name:
                    Qs = get_WSLSS_Qs(2, n_subj)[0]
                    Qs = theano.shared(np.array(Qs, dtype='float32'))  # initialize at WSLSS Qs
                    if verbose:
                        print("Initializing Qs at WSLSS.")
                else:
                    Qs = 0.5 * T.ones((n_subj, 2, 2, 2, 2, 2), dtype='float32')  # (n_subj, prev_prev_choice, prev_prev_reward, prev_choice, prev_reward, choice)
            elif 'S' in model_name:  # S models
                if 'i' in model_name:
                    Qs = get_WSLS_Qs(2, n_subj)[0]
                    Qs = theano.shared(np.array(Qs, dtype='float32'))  # initialize at WSLS Qs
                    if verbose:
                        print("Initializing Qs at WSLS.")
                else:
                    Qs = 0.5 * T.ones((n_subj, 2, 2, 2), dtype='float32')  # (n_subj, prev_choice, prev_reward, choice)
            elif 'ab' in model_name:  # letter models
                Qs = 0.5 * T.ones((n_subj, 2), dtype='float32')
            _ = T.ones(n_subj, dtype='float32')

            # Calculate Q-values for all trials (RL models only)
            if n_trials_back == 0:
                [Qs, _], _ = theano.scan(  # shape: (n_trials-2, n_subj, prev_choice); starts predicting at trial 3!
                    fn=update_Q_0,
                    sequences=[
                        choices[0:-2], rewards[0:-2],  # prev_prev_choice, prev_prev_reward (state in SS models)
                        choices[1:-1], rewards[1:-1],  # prev_choice, prev_reward (state in S and SS models)
                        choices[2:], rewards[2:]],  # this choice (action) is updated based on this reward
                    outputs_info=[Qs, _],
                    non_sequences=[alpha, nalpha, calpha, cnalpha, n_subj])

            elif n_trials_back == 1:
                [Qs, _], _ = theano.scan(  # shape: (n_trials-2, n_subj, prev_choice, prev_reward, choice)
                    fn=update_Q_1,
                    sequences=[
                        choices[0:-2], rewards[0:-2],
                        choices[1:-1], rewards[1:-1],
                        choices[2:], rewards[2:]],
                    outputs_info=[Qs, _],
                    non_sequences=[alpha, nalpha, calpha, cnalpha, n_subj])

            elif n_trials_back == 2:
                [Qs, _], _ = theano.scan(  # shape: (n_trials-2, n_subj, prev_prev_choice, prev_prev_reward, prev_choice, prev_reward, choice)
                    fn=update_Q_2,
                    sequences=[
                        choices[0:-2], rewards[0:-2],
                        choices[1:-1], rewards[1:-1],
                        choices[2:], rewards[2:]],
                    outputs_info=[Qs, _],
                    non_sequences=[alpha, nalpha, calpha, cnalpha, n_subj])

        if 'RL' in model_name or 'WSLS' in model_name:

            # Initialize p_right for a single trial
            p_right = 0.5 * T.ones(n_subj, dtype='float32')  # shape: (n_subj)

            # Translate Q-values into probabilities for all trials
            if n_trials_back == 0:
                p_right, _ = theano.scan(  # shape: (n_trials-2, n_subj)
                    fn=p_from_Q_0,
                    sequences=[Qs,
                               choices[0:-2], rewards[0:-2],  # prev_prev_choice, prev_prev_reward (state in SS models)
                               choices[1:-1], rewards[1:-1]  # prev_choice, prev_reward (state in S and SS models)
                               ],
                    outputs_info=[p_right],
                    non_sequences=[n_subj, beta, persev])

            elif n_trials_back == 1:
                p_right, _ = theano.scan(  # shape: (n_trials-2, n_subj)
                    fn=p_from_Q_1,
                    sequences=[Qs,
                               choices[0:-2], rewards[0:-2],
                               choices[1:-1], rewards[1:-1]
                               ],
                    outputs_info=[p_right],
                    non_sequences=[n_subj, beta, persev])

            elif n_trials_back == 2:
                p_right, _ = theano.scan(  # shape: (n_trials-2, n_subj)
                    fn=p_from_Q_2,
                    sequences=[Qs,
                               choices[0:-2], rewards[0:-2],
                               choices[1:-1], rewards[1:-1]
                               ],
                    outputs_info=[p_right],
                    non_sequences=[n_subj, beta, persev])

        elif 'B' in model_name:

            lik_cor, lik_inc = get_likelihoods(rewards, choices, p_reward, p_noisy)

            # Get posterior & calculate probability of subsequent trial
            p_r = 0.5 * T.ones(n_subj, dtype='float32')
            [p_r, p_right], _ = theano.scan(fn=post_from_lik,  # shape (n_trials, n_subj); starts predicting at trial 1!
                                            sequences=[lik_cor, lik_inc, scaled_persev_bonus],
                                            outputs_info=[p_r, None],
                                            non_sequences=[p_switch, beta])
            p_right = p_right[2:]  # predict from trial 3 onward, not trial 1 (consistent with RL / strat models)

        # Use Bernoulli to sample responses
        model_choices = pm.Bernoulli('model_choices', p=p_right[:-1], observed=choices[3:])  # predict from trial 3 on; discard last p_right because there is no trial to predict after the last value update

        # Check model logp and RV logps (will crash if they are nan or -inf)
        if verbose or print_logps:
            print_logp_info(model)
            theano.printing.Print('all choices')(choices)
            theano.printing.Print('all rewards')(rewards)
            if 'RL' in model_name:
                theano.printing.Print('predicting Qs[:-1]')(Qs[:-1])
            theano.printing.Print('predicting p_right[:-1]')(p_right[:-1])
            theano.printing.Print('predicted choices[3:]')(choices[3:])

    return model, n_params, n_trials, save_dir, save_id


def fit_model(model, n_params, n_subj, n_trials,
              save_dir, save_id,
              fit_mcmc=False, fit_map=True,
              n_samples=100, n_tune=100, n_chains=1, n_cores=2, target_accept=0.8):

    # Sample the model
    if fit_mcmc:
        trace = pm.sample(n_samples, model=model, tune=n_tune, chains=n_chains, cores=n_cores, nuts_kwargs=dict(target_accept=target_accept))
    if fit_map:
        map, opt_result = pm.find_MAP(model=model, return_raw=True)  # default: method='L-BFGS-B'

    # Get results
    if fit_mcmc:
        model_summary = pm.summary(trace)
        waic = pm.waic(trace, model)
        print("MCMC estimates: {0}\nWAIC: {1}".format(model_summary, waic.WAIC))

    if fit_map:
        nll = opt_result['fun']
        # bic = np.log(n_subj * n_trials) * n_params + 2 * nll
        # should be:
        bic = np.log(n_trials) * n_subj * n_params + 2 * nll
        print("NLL: {0},\nBIC: {1}".format(nll, bic))

    # Save results
    if fit_mcmc:
        print('Saving trace, trace plot, model, and model summary to {0}{1}...\n'.format(save_dir, save_id))
        with open(save_dir + save_id + '.pickle', 'wb') as handle:
            pickle.dump({'trace': trace, 'model': model, 'summary': model_summary, 'WAIC': waic.WAIC},
                        handle, protocol=pickle.HIGHEST_PROTOCOL)
    if fit_map:
        print('Saving map estimate, nll, and bic to {0}{1}...\n'.format(save_dir, save_id))
        with open(save_dir + save_id + '.pickle', 'wb') as handle:
            pickle.dump({'map': map, 'nll': nll, 'bic': bic},
                        handle, protocol=pickle.HIGHEST_PROTOCOL)

    if fit_map:
        return nll, bic


# GET LIST OF MODELS TO RUN
# model_names = ['RLab' + i for i in
#                ['', 'c', 'n', 'p', 'SS', 'SSi', 'S', 'Si', 'cnxp', 'cnxpS', 'cnxpSi', 'cnxpSS', 'cnxpSSi',
#                 'cn', 'cnx']]

model_names = []

# Add RL models
for c in ['', 'c']:
    for n in ['', 'n']:
        for x in ['', 'x']:
            for p in ['', 'p']:
                for S in ['', 'S', 'SS']:
                    model_names.append('RLab' + c + n + x + p + S)
                    if S == 'S' or S == 'SS':
                        model_names.append('RLab' + c + n + x + p + S + 'i')

# Add strategy models
model_names.extend(['WSLS', 'WSLSS'])

# Add Bayesian models
for s in ['', 's']:
    for p in ['', 'p']:
        for r in ['', 'r']:
            model_names.append('Bb' + s + p + r)

print("Getting ready to run {0} models: {1}".format(len(model_names), model_names))

# Load behavioral data on which to run the model(s)
n_subj, rewards, choices, group, n_groups, age_z = load_data(run_on_cluster, n_subj='all')

# Run all models
nll_bics = pd.DataFrame()
for model_name in model_names:

    # Create model
    model, n_params, n_trials, save_dir, save_id = create_model(
        choices=choices, rewards=rewards, group=group, model_name=model_name, n_subj=n_subj, n_trials='all', verbose=False)

    # Fit parameters, calculate model fits, make plots, and save everything
    nll, bic = fit_model(model, n_params, n_subj, n_trials, save_dir, save_id)

    nll_bics = nll_bics.append([[model_name, n_params, nll, bic]])
    nll_bics.to_csv(save_dir + '/nll_bics_temp.csv', index=False)

nll_bics.columns = ['model_name', 'n_params', 'nll', 'bic']
nll_bics.to_csv(save_dir + '/nll_bics.csv', index=False)
