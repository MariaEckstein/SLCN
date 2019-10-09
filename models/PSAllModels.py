run_on_cluster = False
save_dir_appx = 'new_ML_models/'
import itertools

# GET LIST OF MODELS TO RUN
# # List of selected models
# slope_appxs_abcnp = []
# slope_appxs_abcxp = []
# slope_appxs_bspr = []
#
# for i in range(6):
#     slope_appxs_abcnp.extend([''.join(i) for i in list(itertools.combinations('lyoqt', i))])
#     slope_appxs_abcxp.extend([''.join(i) for i in list(itertools.combinations('lyout', i))])
# for i in range(5):
#     slope_appxs_bspr.extend([''.join(i) for i in list(itertools.combinations('ywtv', i))])
#
# model_names = ['RLabcnp' + appx for appx in slope_appxs_abcnp]
# model_names.extend(['Bbspr' + appx for appx in slope_appxs_bspr])
# model_names.extend(['RLabcxp' + appx for appx in slope_appxs_abcxp])

# # model_names = ['RLabcnplyoqt', 'RLabcxplyout']
# # model_names = ['WSLS', 'WSLSy', 'WSLSS', 'WSLSy']
# model_names = ['Bbsprywvt', 'RLabcnplyoqt'] #  ['Bbsrywv']
# model_names = ['RLab', 'RLabc', 'RLabcn', 'RLabcnp']
# model_names = ['RLabcnp']
# model_names = [
#     # 'RLabcp', 'RLabcpn', 'RLabxp', 'RLabcpx', 'RLab', 'RLabc',
#     # 'Bbpr', 'Bbspr', 'Bbsp', 'Bb', 'Bbs',
#     # 'Bbsprywtv', 'Bbspry', 'Bbspryw', 'Bbsprt', 'Bbsprv',
#     # 'Bbspr', 'Bbsprywtv',
#     'RLabcpnlyoqt', 'RLabcpnl', 'RLabcpny', 'RLabcpno', 'RLabcpnq', 'RLabcpnt',
#     # 'WSLS', 'WSLSS'
# ]

model_names = [
    'RLabnp2lqyt'
    # 'RLabcnplyoqt',
    # 'RLabnp2',
    # 'RLab', 'RLabc', 'RLabcp', 'RLabcpn', 'RLabcnpx',
    # 'RLabcxnplyoqtu',
    # 'RLabcxnp',
    # 'Bbspry',
    # 'Bbsprywtv',
    # 'RLab', 'RLabcxpS', 'RLabcxpSi', 'RLabcxpSS', 'RLabcxpSSi', 'RLabcxpSm',
    # 'WSLS', 'WSLSS',
    # 'Bbspr', 'Bbpr', 'Bbp', 'Bb', 'B',
    # 'RLabcnpx', 'Bbpr',
]

# # All possible models
# model_names = []
#
# # Add RL models
# for c in ['', 'c']:
#     for n in ['', 'n']:
#         for x in ['', 'x']:
#             for p in ['', 'p']:
#                 for S in ['']:  # , 'S', 'SS']:  # , 'SSS'
#                     model_names.append('RLab' + c + n + x + p + S)
#                     if S == 'S' or S == 'SS':
#                         model_names.append('RLab' + c + n + x + p + S + 'i')
# #
# # # Add strategy models
# # model_names.extend(['WSLS', 'WSLSS'])
#
# # Add Bayesian models
# for s in ['', 's']:
#     for p in ['', 'p']:
#         for r in ['', 'r']:
#             model_names.append('Bb' + s + p + r)

print("Getting ready to run {0} models: {1}".format(len(model_names), model_names))

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import pymc3 as pm
import seaborn as sns
import theano
import theano.tensor as T
from shared_modeling_simulation import *
from modeling_helpers import load_data, get_save_dir_and_save_id, print_logp_info

floatX = 'float32'
theano.config.floatX = 'float32'
theano.config.warn_float64 = 'warn'


def get_slope(param_name, sd_testval, slope_testval, sd=10):

    if n_groups > 1:
        param_slope_mu = pm.Normal(param_name + '_slope_mu', mu=0, sd=sd, shape=1, testval=slope_testval * T.ones(1, dtype='int32'))
        param_slope_sd = pm.HalfNormal(param_name + '_slope_sd', sd=sd, shape=1, testval=sd_testval * T.ones(1, dtype='int32'))
    else:
        param_slope_mu, param_slope_sd = 0, sd

    return pm.Normal(param_name + '_slope', mu=param_slope_mu, sd=param_slope_sd, shape=n_groups, testval=slope_testval * T.ones(n_groups, dtype='int32'))


def get_sd(param_name, sd_testval, sd=10):

    if n_groups > 1:
        param_sd_sd = pm.HalfNormal(param_name + '_sd_sd', sd=sd, shape=1, testval=sd_testval * T.ones(1, dtype='int32'))
    else:
        param_sd_sd = sd

    return pm.HalfNormal(param_name + '_sd', sd=param_sd_sd, shape=n_groups, testval=sd_testval * T.ones(n_groups, dtype='int32'))


def get_int(param_name, distribution, upper, int_testval, n_groups, sd=10):

    if (distribution == 'Beta') or (  # alpha, nalpha, calpha_sc, cnalpha_sc, p_switch, p_reward
            distribution == 'Gamma'):  # beta

        if n_groups > 1:
            param_int_a = pm.Uniform(param_name + '_int_a', lower=0, upper=upper)
            param_int_b = pm.Uniform(param_name + '_int_b', lower=0, upper=upper)
        else:
            param_int_a, param_int_b = 1, 1

        if distribution == 'Beta':
            return pm.Beta(param_name + '_int', alpha=param_int_a, beta=param_int_b, shape=n_groups, testval=int_testval * T.ones(n_groups, dtype='int32'))

        elif distribution == 'Gamma':
            return pm.Gamma(param_name + '_int', alpha=param_int_a, beta=param_int_b, shape=n_groups, testval=int_testval * T.ones(n_groups, dtype='int32'))

    elif distribution == 'Normal':  # persev

        if n_groups > 1:
            param_int_mu = pm.Normal(param_name + '_int_mu', mu=0, sd=sd, shape=1)
            param_int_sd = pm.HalfNormal(param_name + '_int_sd', sd=sd, shape=1)
        else:
            param_int_mu, param_int_sd = 0, sd

        return pm.Normal(param_name + '_int', mu=param_int_mu, sd=param_int_sd, shape=n_groups, testval=int_testval * T.ones(n_groups, dtype='int32'))


def get_a_b(param_name, upper, n_groups):

    if n_groups > 1:
        param_a_a = pm.Uniform(param_name + '_a_a', lower=0, upper=upper)
        param_a_b = pm.Uniform(param_name + '_a_b', lower=0, upper=upper)
        param_b_a = pm.Uniform(param_name + '_b_a', lower=0, upper=upper)
        param_b_b = pm.Uniform(param_name + '_b_b', lower=0, upper=upper)
    else:
        param_a_a, param_a_b, param_b_a, param_b_b = 1, 1, 1, 1

    param_a = pm.Gamma(param_name + '_a', alpha=param_a_a, beta=param_a_b, shape=n_groups)
    param_b = pm.Gamma(param_name + '_b', alpha=param_b_a, beta=param_b_b, shape=n_groups)

    return param_a, param_b


def get_mu_sd(param_name, n_groups, sd=10):

    if n_groups > 1:
        param_mu_mu = pm.Normal(param_name + '_mu_mu', mu=0, sd=sd)
        param_mu_sd = pm.HalfNormal(param_name + '_mu_sd', sd=sd)
        param_sd_sd = pm.HalfNormal(param_name + '_sd_sd', sd=sd)
    else:
        param_mu_mu, param_mu_sd, param_sd_sd = 0, sd, sd

    param_mu = pm.Normal(param_name + '_mu', mu=param_mu_mu, sd=param_mu_sd, shape=n_groups)
    param_sd = pm.HalfNormal(param_name + '_sd', sd=param_sd_sd, shape=n_groups)

    return param_mu, param_sd


def create_parameter(param_name, distribution, fit_slope, n_groups, group, n_subj, upper, slope_variable, contrast,
                     sd_testval=0.5, slope_testval=0, int_testval=0.5, param_testval=0.5):

    print("Adding free parameter {0}".format(param_name))

    if fit_slope:
        param_sd = get_sd(param_name, sd_testval)

        if np.any([contrast == c for c in ['linear', 'quadratic', 'cubic', '4th']]):
            param_slope = get_slope(param_name, sd_testval, slope_testval)
            print("Adding slope for {}.".format(param_name))
        else:
            param_slope = T.zeros(n_groups)

        if np.any([contrast == c for c in ['quadratic', 'cubic', '4th']]):
            param_slope2 = get_slope(param_name + '_2', sd_testval, slope_testval)
            print("Adding quadratic slope for {}.".format(param_name))
        else:
            param_slope2 = T.zeros(n_groups)

        if np.any([contrast == c for c in ['cubic', '4th']]):
            param_slope3 = get_slope(param_name + '_3', sd_testval, slope_testval)
            print("Adding cubic slope for {}.".format(param_name))
        else:
            param_slope3 = T.zeros(n_groups)

        if np.any([contrast == c for c in ['4th']]):
            param_slope4 = get_slope(param_name + '_4', sd_testval, slope_testval)
            print("Adding 4th-order slope for {}.".format(param_name))
        else:
            param_slope4 = T.zeros(n_groups)

        param_int = get_int(param_name, distribution, upper, int_testval, n_groups)

        if distribution == 'Beta':
            return pm.Bound(pm.Normal, lower=0, upper=1)(param_name, shape=n_subj, testval=param_testval * T.ones(n_subj, dtype='int32'),
                                      mu=param_int[group] + param_slope[group] * slope_variable + param_slope2[group] * T.sqr(slope_variable) + param_slope3[group] * slope_variable * T.sqr(slope_variable) + param_slope4[group] * T.sqr(slope_variable) * T.sqr(slope_variable),
                                      sd=param_sd[group])

        elif distribution == 'Gamma':
            return pm.Bound(pm.Normal, lower=0)(param_name, shape=n_subj, testval=param_testval * T.ones(n_subj, dtype='int32'),
                                      mu=param_int[group] + param_slope[group] * slope_variable + param_slope2[group] * T.sqr(slope_variable) + param_slope3[group] * slope_variable * T.sqr(slope_variable) + param_slope4[group] * T.sqr(slope_variable) * T.sqr(slope_variable),
                                      sd=param_sd[group])

        else:
            return pm.Normal(param_name, shape=n_subj, testval=param_testval * T.ones(n_subj, dtype='int32'),
                             mu=param_int[group] + param_slope[group] * slope_variable + param_slope2[group] * T.sqr(slope_variable) + param_slope3[group] * slope_variable * T.sqr(slope_variable) + param_slope4[group] * T.sqr(slope_variable) * T.sqr(slope_variable),
                             sd=param_sd[group])

    else:
        if distribution == 'Normal':
            param_mu, param_sd = get_mu_sd(param_name, n_groups)
            return pm.Normal(param_name, mu=param_mu[group], sd=param_sd[group], shape=n_subj, testval=param_testval * T.ones(n_subj, dtype='float32'))

        else:
            param_a, param_b = get_a_b(param_name, upper, n_groups)

            if distribution == 'Beta':
                return pm.Beta(param_name, alpha=param_a[group], beta=param_b[group], shape=n_subj, testval=param_testval * T.ones(n_subj, dtype='float32'))

            elif distribution == 'Gamma':
                return pm.Gamma(param_name, alpha=param_a[group], beta=param_b[group], shape=n_subj, testval=param_testval * T.ones(n_subj, dtype='float32'))


def create_model(choices, rewards, group, age,
                 n_subj='all', n_trials='all',  # 'all' or int
                 model_name='ab',  # ab, abc, abn, abcn, abcnx, abcnS, abcnSS, WSLS, WSLSS, etc. etc.
                 slope_variable='age_z', contrast='linear',
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
        # persev_bonus = np.concatenate([np.zeros((1, n_subj)), persev_bonus])  # add 0 bonus for first trial
        persev_bonus = theano.shared(np.asarray(persev_bonus, dtype='int16'))

    # Transform everything into theano.shared variables
    rewards = theano.shared(np.asarray(rewards, dtype='int16'))
    choices = theano.shared(np.asarray(choices, dtype='int16'))
    group = theano.shared(np.asarray(group, dtype='int16'))
    slope_variable = theano.shared(np.asarray(age[slope_variable], dtype='float32'))

    # Get n_trials_back, n_params, and file_name_suff
    n_trials_back = get_n_trials_back(model_name)
    n_params = get_n_params(model_name, n_subj, n_groups, contrast=contrast)
    file_name_suff = model_name + ''
    save_dir, save_id = get_save_dir_and_save_id(run_on_cluster, file_name_suff, fitted_data_name, n_samples)
    save_dir += save_dir_appx
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(save_dir + 'plots/')
    print("Working on model '{0}', which has {1} free parameters. Save_dir: {2}".format(model_name, n_params, save_dir))

    # Get fixed Q-values for WSLS and WSLSS
    if 'WSLSS' in model_name:  # "stay unless you fail to receive reward twice in a row for the same action."
        Qs = get_WSLSS_Qs(n_trials, n_subj)
        Qs = theano.shared(np.asarray(Qs, dtype='float32'))

    elif 'WSLS' in model_name:  # "stay if you won; switch if you lost"
        Qs = get_WSLS_Qs(n_trials, n_subj)
        Qs = theano.shared(np.asarray(Qs, dtype='float32'))

    print("Compiling models for {0} with {1} samples and {2} tuning steps...\n".format(fitted_data_name, n_samples, n_tune))
    with pm.Model() as model:
        if not fit_individuals:

            # RL, Bayes, and WSLS
            if ('b' in model_name) or ('WSLS' in model_name):
                beta = create_parameter('beta', 'Gamma', 'y' in model_name, n_groups, group, n_subj, upper, slope_variable, contrast)
                                        # sd_testval=1.5, slope_testval=0.6, int_testval=3.8, param_testval=3)  # RLabcnplyoqt beta mcmc results 2019-06-19
            else:
                beta = pm.Gamma('beta', alpha=1, beta=1, shape=n_subj)  # won't be used - necessary for sampling
                print("This model does not have beta.")

            if 'p' in model_name:
                persev = create_parameter('persev', 'Normal', 't' in model_name, n_groups, group, n_subj, upper, slope_variable, contrast)
                                          # sd_testval=0.11, slope_testval=-0.03, int_testval=-0.01, param_testval=0.01, sd=1)  # RLabcnplyoqt persev mcmc results 2019-06-19
            else:
                persev = pm.Deterministic('persev', T.zeros(n_subj, dtype='float32'))
                print("Setting persev = 0.")

            if 'RL' in model_name:
                if 'a' in model_name:
                    alpha = create_parameter('alpha', 'Beta', 'l' in model_name, n_groups, group, n_subj, upper, slope_variable, contrast)
                                                # sd_testval=1.2, slope_testval=0.3, int_testval=0.7, param_testval=0.7)  # RLabcnplyoqt alpha mcmc results 2019-06-19
                else:
                    alpha = pm.Deterministic('alpha', T.ones(n_subj, dtype='float32'))
                    print("Setting alpha = 1")

                if 'n' in model_name:
                    nalpha = create_parameter('nalpha', 'Beta', 'q' in model_name, n_groups, group, n_subj, upper, slope_variable, contrast)
                                                 # sd_testval=0.95, slope_testval=-0.2, int_testval=0.65, param_testval=0.65)  # RLabcnplyoqt nalpha mcmc results 2019-06-19
                else:
                    nalpha = pm.Deterministic('nalpha', 1 * alpha)
                    print("Setting nalpha = alpha.")

                if 'c' in model_name:
                    calpha = create_parameter('calpha', 'Beta', 'o' in model_name, n_groups, group, n_subj, upper, slope_variable, contrast)
                                                    # sd_testval=1.8, slope_testval=-0.3, int_testval=0, param_testval=0)  # RLabcnplyoqt calpha_sc mcmc results 2019-06-19
                elif '2' in model_name:
                    calpha = alpha.copy()
                    print("Setting calpha = alpha.")
                else:
                    calpha = pm.Deterministic('calpha', T.zeros(n_subj, dtype='float32'))
                    print("Setting calpha = 0.")
                # calpha = pm.Deterministic('calpha', alpha * calpha_sc)

                if 'x' in model_name:
                    cnalpha = create_parameter('cnalpha', 'Beta', 'u' in model_name, n_groups, group, n_subj, upper, slope_variable, contrast)
                                                     # sd_testval=0.3, slope_testval=0.3, int_testval=0, param_testval=0)  # RLabcnplyoqt cnalpha mcmc results 2019-06-19
                elif '2' in model_name:
                    cnalpha = nalpha.copy()
                    print("Setting cnalpha = nalpha.")
                else:
                    cnalpha = pm.Deterministic('cnalpha', T.zeros(n_subj, dtype='float32'))

                if 'm' in model_name:
                    m = create_parameter('m', 'Beta', 'z' in model_name, n_groups, group, n_subj, upper, slope_variable, contrast)
                else:
                    m = pm.Deterministic('m', T.ones(n_subj, dtype='float32'))
                    print("Setting m = 1.")

            elif 'B' in model_name:

                p_noisy = 1e-5 * T.as_tensor_variable(1)

                if 's' in model_name:
                    p_switch = create_parameter('p_switch', 'Beta', 'w' in model_name, n_groups, group, n_subj, upper, slope_variable, contrast)
                                                # sd_testval=0.1, slope_testval=-0.1, int_testval=0.3, param_testval=0.3)
                else:
                    p_switch = pm.Deterministic('p_switch', 0.05081582 * T.ones(n_subj, dtype='float32'))
                    print("Setting p_switch to 0.0508...")

                if 'r' in model_name:
                    p_reward = create_parameter('p_reward', 'Beta', 'v' in model_name, n_groups, group, n_subj, upper, slope_variable, contrast)
                                                # sd_testval=1, slope_testval=-0.3, int_testval=0.5, param_testval=0.5)
                else:
                    p_reward = pm.Deterministic('p_reward', 0.75 * T.ones(n_subj, dtype='float32'))
                    print("Setting p_reward to 0.75.")

        else:  # if fit_individuals == True:

            # beta = pm.Gamma('beta', alpha=1, beta=1, shape=n_subj)
            #
            # if 'p' in model_name:
            #     persev = pm.Normal('persev', mu=0, sd=1, shape=n_subj, testval=0.01 * T.ones(n_subj))
            # else:
            #     persev = pm.Deterministic('persev', T.zeros(n_subj, dtype='float32'))
            #     print("Setting persev = 0.")
            #
            # if 'RL' in model_name:
            #     if 'a' in model_name:
            #         alpha = pm.Beta('alpha', alpha=1, beta=1, shape=n_subj, testval=0.8 * T.ones(n_subj))
            #     else:
            #         alpha = pm.Deterministic('alpha', T.ones(n_subj, dtype='float32'))
            #         print("Setting alpha = 1.")
            #
            #     if 'n' in model_name:
            #         nalpha = pm.Beta('nalpha', alpha=1, beta=1, shape=n_subj, testval=0.8 * T.ones(n_subj))
            #     else:
            #         nalpha = pm.Deterministic('nalpha', 1 * alpha)
            #         print("Setting nalpha = alpha.")
            #
            #     if 'c' in model_name:
            #         calpha_sc = pm.Beta('calpha_sc', alpha=1, beta=1, shape=n_subj, testval=0.8 * T.ones(n_subj))
            #     else:
            #         calpha_sc = 0
            #         print("Setting calpha_sc = 0.")
            #
            #     if 'x' in model_name:
            #         cnalpha_sc = pm.Beta('cnalpha_sc', alpha=1, beta=1, shape=n_subj, testval=0.8 * T.ones(n_subj))
            #     elif 'c' in model_name:
            #         cnalpha_sc = pm.Deterministic('cnalpha_sc', 1 * calpha_sc)
            #         print("Setting cnalpha_sc = calpha_sc.")
            #     else:
            #         cnalpha_sc = 0
            #         print("Setting cnalpha_sc = 0.")
            #
            #     if 'm' in model_name:
            #         m = pm.Beta('m', alpha=1, beta=1, shape=n_subj, testval=0.8 * T.ones(n_subj))
            #     else:
            #         m = pm.Deterministic('m', 0 * alpha)
            #         print("Setting m = 0.")
            #
            #     calpha = pm.Deterministic('calpha', alpha * calpha_sc)
            #     cnalpha = pm.Deterministic('cnalpha', nalpha * cnalpha_sc)
            #
            # elif 'B' in model_name:
            #     p_noisy = pm.Deterministic('p_noisy', 1e-5 * T.ones(n_subj, dtype='float32'))
            #     if 's' in model_name:
            #         p_switch = pm.Beta('p_switch', alpha=1, beta=1, shape=n_subj, testval=0.1 * T.ones(n_subj))
            #     else:
            #         p_switch = pm.Deterministic('p_switch', 0.05081582 * T.ones(n_subj))  # checked on 2019-06-03 in R: `mean(all_files$switch_trial)`
            #         print("Setting p_switch = 0.05081582.")
            #
            #     if 'r' in model_name:
            #         p_reward = pm.Beta('p_reward', alpha=1, beta=1, shape=n_subj, testval=0.8 * T.ones(n_subj))
            #     else:
            #         p_reward = pm.Deterministic('p_reward', 0.75 * T.ones(n_subj))  # 0.75 because p_reward is the prob. of getting reward if choice is correct
            #         print("Setting p_reward = 0.75.")

            if 'y' in model_name:  # model with slope
                beta_intercept = pm.Gamma('beta_intercept', alpha=1, beta=1, shape=n_groups)  # , testval=0.5 * T.ones(n_groups, dtype='int32'))
                beta_slope = pm.Uniform('beta_slope', lower=-10, upper=10, shape=n_groups, testval=0.5 * T.ones(n_groups, dtype='int32'))
                if contrast == 'quadratic':
                    beta_slope2 = pm.Uniform('beta_slope2', lower=-1, upper=1, shape=n_groups)  # , testval=0.1 * T.ones(n_groups, dtype='int32'))
                else:
                    beta_slope2 = T.zeros(n_groups, dtype='int16')
                beta = pm.Deterministic('beta', beta_intercept[group] + beta_slope[group] * slope_variable + beta_slope2[group] * T.sqr(slope_variable))
                print("Drawing slope, intercept, and noise for beta.")
            else:
                # beta = pm.Gamma('beta', alpha=1, beta=1, shape=n_subj, testval=2 * T.ones(n_subj, dtype='float32'))
                beta = pm.Uniform('beta', lower=0, upper=15, shape=n_subj, testval=2 * T.ones(n_subj, dtype='float32'))
            print("Adding free parameter beta.")

            if 'p' in model_name:
                if 't' in model_name:
                    persev_intercept = pm.Uniform('persev_intercept', lower=-1, upper=1, shape=n_groups, testval=0.1 * T.ones(n_groups, dtype='int32'))
                    persev_slope = pm.Uniform('persev_slope', lower=-1, upper=1, shape=n_groups, testval=-0.1 * T.ones(n_groups, dtype='int32'))
                    if contrast == 'quadratic':
                        persev_slope2 = pm.Uniform('persev_slope2', lower=-1, upper=1, shape=n_groups, testval=-0.1 * T.ones(n_groups, dtype='int32'))
                    else:
                        persev_slope2 = T.zeros(n_groups, dtype='int16')
                    persev = pm.Deterministic('persev', persev_intercept[group] + persev_slope[group] * slope_variable + persev_slope2[group] * T.sqr(slope_variable))
                    print("Drawing slope, intercept, and noise for persev.")
                else:
                    persev = pm.Uniform('persev', lower=-1, upper=1, shape=n_subj, testval=0 * T.ones(n_subj, dtype='float32'))
                print("Adding free parameter persev.")
            else:
                persev = pm.Deterministic('persev', T.zeros(n_subj, dtype='float32'))

            if 'RL' in model_name:
                if 'a' in model_name:
                    if 'l' in model_name:
                        alpha_intercept = pm.Beta('alpha_intercept', alpha=1, beta=1, shape=n_groups, testval=0.75 * T.ones(n_groups, dtype='int32'))
                        alpha_slope = pm.Uniform('alpha_slope', lower=-1, upper=1, shape=n_groups, testval=0.1 * T.ones(n_groups, dtype='int32'))
                        if contrast == 'quadratic':
                            alpha_slope2 = pm.Uniform('alpha_slope2', lower=-1, upper=1, shape=n_groups, testval=0.1 * T.ones(n_groups, dtype='int32'))
                        else:
                            alpha_slope2 = T.zeros(n_groups, dtype='int16')
                        alpha_unbound = pm.Deterministic('alpha_unbound', alpha_intercept[group] + alpha_slope[group] * slope_variable + alpha_slope2[group] * T.sqr(slope_variable))
                        alpha = pm.Deterministic('alpha', T.nnet.sigmoid(alpha_unbound))  # element-wise sigmoid: sigmoid(x) = \frac{1}{1 + \exp(-x)}
                        print("Drawing slope, intercept, and noise for alpha.")
                    else:
                        alpha = pm.Beta('alpha', alpha=1, beta=1, shape=n_subj, testval=0.5 * T.ones(n_subj, dtype='float32'))
                    print("Adding free parameter alpha.")
                else:
                    alpha = pm.Deterministic('alpha', T.ones(n_subj, dtype='float32'))
                    print("Setting alpha = 1.")

                if 'n' in model_name:
                    if 'q' in model_name:
                        nalpha_intercept = pm.Beta('nalpha_intercept', alpha=1, beta=1, shape=n_groups, testval=0.5 * T.ones(n_groups, dtype='int32'))
                        nalpha_slope = pm.Uniform('nalpha_slope', lower=-1, upper=1, shape=n_groups, testval=0.1 * T.ones(n_groups, dtype='int32'))
                        if contrast == 'quadratic':
                            nalpha_slope2 = pm.Uniform('nalpha_slope2', lower=-1, upper=1, shape=n_groups, testval=0.1 * T.ones(n_groups, dtype='int32'))
                        else:
                            nalpha_slope2= T.zeros(n_groups, dtype='int16')
                        nalpha_unbound = pm.Deterministic('nalpha_unbound', nalpha_intercept[group] + nalpha_slope[group] * slope_variable + nalpha_slope2[group] * T.sqr(slope_variable))
                        nalpha = pm.Deterministic('nalpha', T.nnet.sigmoid(nalpha_unbound))
                        print("Drawing slope, intercept, and noise for nalpha.")
                    else:
                        nalpha = pm.Beta('nalpha', alpha=1, beta=1, shape=n_subj, testval=0.5 * T.ones(n_subj, dtype='float32'))
                    print("Adding free parameter nalpha.")
                else:
                    nalpha = pm.Deterministic('nalpha', 1 * alpha)
                    print("Setting nalpha = alpha.")

                if 'c' in model_name:
                    if 'o' in model_name:
                        calpha_intercept = pm.Beta('calpha_intercept', alpha=1, beta=1, shape=n_groups, testval=0.5 * T.ones(n_groups, dtype='int32'))
                        calpha_slope = pm.Uniform('calpha_slope', lower=-1, upper=1, shape=n_groups, testval=-0.1 * T.ones(n_groups, dtype='int32'))
                        if contrast == 'quadratic':
                            calpha_slope2 = pm.Uniform('calpha_slope2', lower=-1, upper=1, shape=n_groups, testval=-0.1 * T.ones(n_groups, dtype='int32'))
                        else:
                            calpha_slope2= T.zeros(n_groups, dtype='int16')
                        calpha_unbound = pm.Deterministic('calpha_unbound', calpha_intercept[group] + calpha_slope[group] * slope_variable + calpha_slope2[group] * T.sqr(slope_variable))
                        calpha = pm.Deterministic('calpha', T.nnet.sigmoid(calpha_unbound))
                        print("Drawing slope, intercept, and noise for calpha.")
                    else:
                        calpha = pm.Beta('calpha', alpha=1, beta=1, shape=n_subj, testval=0.5 * T.ones(n_subj, dtype='float32'))
                    print("Adding free parameter calpha.")

                elif '2' in model_name:
                    calpha = pm.Deterministic('calpha', 1 * alpha)
                    print("Setting calpha = alpha")

                else:
                    calpha = 0
                    print("Setting calpha = 0.")

                if 'x' in model_name:
                    if 'u' in model_name:
                        cnalpha_intercept = pm.Beta('cnalpha_intercept', alpha=1, beta=1, shape=n_groups, testval=0.5 * T.ones(n_groups, dtype='int32'))
                        cnalpha_slope = pm.Uniform('cnalpha_slope', lower=-1, upper=1, shape=n_groups, testval=-0.1 * T.ones(n_groups, dtype='int32'))
                        if contrast == 'quadratic':
                            cnalpha_slope2 = pm.Uniform('cnalpha_slope2', lower=-1, upper=1, shape=n_groups, testval=-0.1 * T.ones(n_groups, dtype='int32'))
                        else:
                            cnalpha_slope2= T.zeros(n_groups, dtype='int16')
                        cnalpha_unbound = pm.Deterministic('cnalpha_unbound', cnalpha_intercept[group] + cnalpha_slope[group] * slope_variable + cnalpha_slope2[group] * T.sqr(slope_variable))
                        cnalpha = pm.Deterministic('cnalpha', T.nnet.sigmoid(cnalpha_unbound))
                        print("Drawing slope, intercept, and noise for cnalpha.")

                    else:
                        cnalpha = pm.Beta('cnalpha', alpha=1, beta=1, shape=n_subj, testval=0.5 * T.ones(n_subj, dtype='float32'))
                    print("Adding free parameter cnalpha.")
                # elif 'c' in model_name:
                #     cnalpha = pm.Deterministic('cnalpha', 1 * calpha)
                #     print("Setting cnalpha = calpha.")

                elif '2' in model_name:
                    cnalpha = pm.Deterministic('cnalpha', 1 * nalpha)
                    print("Setting cnalpha = nalpha")

                else:
                    cnalpha = 0
                    print("Setting cnalpha = 0.")

                if 'm' in model_name:
                    if 'z' in model_name:
                        m_intercept = pm.Beta('m_intercept', alpha=1, beta=1, shape=n_groups, testval=0.5 * T.ones(n_groups, dtype='int32'))
                        m_slope = pm.Uniform('m_slope', lower=-1, upper=1, shape=n_groups, testval=-0.1 * T.ones(n_groups, dtype='int32'))
                        if contrast == 'quadratic':
                            m_slope2 = pm.Uniform('m_slope2', lower=-1, upper=1, shape=n_groups, testval=-0.1 * T.ones(n_groups, dtype='int32'))
                        else:
                            m_slope2= T.zeros(n_groups, dtype='int16')
                        m = pm.Deterministic('m', m_intercept[group] + m_slope[group] * slope_variable + m_slope2[group] * T.sqr(slope_variable))
                        print("Drawing slope, intercept, and noise for m.")
                    else:
                        m = pm.Beta('m', alpha=1, beta=1, shape=n_subj, testval=0.5 * T.ones(n_subj, dtype='float32'))
                    print("Adding free parameter m (multiplier for mirror alpha, nalpha, calpha, cnalpha).")
                else:
                    m = pm.Deterministic('m', 0 * alpha)
                    print("Setting m = 0.")

                # calpha = pm.Deterministic('calpha', alpha * calpha_sc)
                # cnalpha = pm.Deterministic('cnalpha', nalpha * cnalpha_sc)

            elif 'B' in model_name:
                p_noisy = pm.Deterministic('p_noisy', 1e-5 * T.ones(n_subj, dtype='float32'))
                if 's' in model_name:
                    if 'w' in model_name:
                        p_switch_intercept = pm.Beta('p_switch_intercept', alpha=1, beta=1, shape=n_groups, testval=0.05 * T.ones(n_groups, dtype='int32'))
                        p_switch_slope = pm.Uniform('p_switch_slope', lower=-1, upper=1, shape=n_groups, testval=-0.1 * T.ones(n_groups, dtype='int32'))
                        p_switch_unbound = pm.Deterministic('p_switch_unbound', p_switch_intercept[group] + p_switch_slope[group] * slope_variable)
                        p_switch = pm.Deterministic('p_switch', T.nnet.sigmoid(p_switch_unbound))
                        print("Drawing slope, intercept, and noise for p_switch.")
                    else:
                        p_switch = pm.Beta('p_switch', alpha=1, beta=1, shape=n_subj, testval=0.0508 * T.ones(n_subj, dtype='float32'))
                    print("Adding free parameter p_switch.")
                else:
                    p_switch = pm.Deterministic('p_switch', 0.0508 * T.ones(n_subj))  # checked on 2019-06-03 in R: `mean(all_files$switch_trial)`
                    print("Setting p_switch = 0.05081582.")

                if 'r' in model_name:
                    if 'v' in model_name:
                        p_reward_intercept = pm.Beta('p_reward_intercept', alpha=1, beta=1, shape=n_groups, testval=0.75 * T.ones(n_groups, dtype='int32'))
                        p_reward_slope = pm.Uniform('p_reward_slope', lower=-1, upper=1, shape=n_groups, testval=-0.1 * T.ones(n_groups, dtype='int32'))
                        p_reward_unbound = pm.Deterministic('p_reward_unbound', p_reward_intercept[group] + p_reward_slope[group] * slope_variable)
                        p_reward = pm.Deterministic('p_reward', T.nnet.sigmoid(p_reward_unbound))
                        print("Drawing slope, intercept, and noise for p_reward.")
                    else:
                        p_reward = pm.Beta('p_reward', alpha=1, beta=1, shape=n_subj, testval=0.75 * T.ones(n_subj, dtype='float32'))
                    print("Adding free parmeter p_reward.")
                else:
                    p_reward = pm.Deterministic('p_reward', 0.75 * T.ones(n_subj))  # 0.75 because p_reward is the prob. of getting reward if choice is correct
                    print("Setting p_reward = 0.75.")

        # Initialize Q-values (with and without bias, i.e., option 'i')
        if 'RL' in model_name:

            if 'SSS' in model_name:  # SSS models
                print("SSS models do not support initialization, so values are initialized randomly!")
                Qs = 0.5 * T.ones((n_subj, 2, 2, 2, 2, 2, 2, 2), dtype='float32')  # (n_subj, prev_prev_choice, prev_prev_reward, prev_choice, prev_reward, choice)
            elif 'SS' in model_name:  # SS models
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
                    non_sequences=[alpha, nalpha, calpha, cnalpha, m, n_subj])

            elif n_trials_back == 1:
                [Qs, _], _ = theano.scan(  # shape: (n_trials-2, n_subj, prev_choice, prev_reward, choice)
                    fn=update_Q_1,
                    sequences=[
                        choices[0:-2], rewards[0:-2],
                        choices[1:-1], rewards[1:-1],
                        choices[2:], rewards[2:]],
                    outputs_info=[Qs, _],
                    non_sequences=[alpha, nalpha, calpha, cnalpha, m, n_subj])

            elif n_trials_back == 2:
                [Qs, _], _ = theano.scan(  # shape: (n_trials-2, n_subj, prev_prev_choice, prev_prev_reward, prev_choice, prev_reward, choice)
                    fn=update_Q_2,
                    sequences=[
                        choices[0:-2], rewards[0:-2],
                        choices[1:-1], rewards[1:-1],
                        choices[2:], rewards[2:]],
                    outputs_info=[Qs, _],
                    non_sequences=[alpha, nalpha, calpha, cnalpha, m, n_subj])

            elif n_trials_back == 3:
                [Qs, _], _ = theano.scan(  # shape: (n_trials-2, n_subj, prev_prev_choice, prev_prev_reward, prev_choice, prev_reward, choice)
                    fn=update_Q_3,
                    sequences=[
                        choices[0:-3], rewards[0:-3],
                        choices[1:-2], rewards[1:-2],
                        choices[2:-1], rewards[2:-1],
                        choices[3:], rewards[3:]],
                    outputs_info=[Qs, _],
                    non_sequences=[alpha, nalpha, calpha, cnalpha, m, n_subj])

        if 'RL' in model_name or 'WSLS' in model_name:

            # Initialize p_right for a single trial
            p_right = 0.5 * T.ones(n_subj, dtype='float32')  # shape: (n_subj)

            # Translate Q-values into probabilities for all trials
            if n_trials_back == 0:
                p_right, _ = theano.scan(  # shape: (n_trials-2, n_subj)
                    fn=p_from_Q_0,
                    sequences=[Qs,
                               choices[1:-1], rewards[1:-1],  # prev_prev_choice, prev_prev_reward (state in SS models)
                               choices[2:], rewards[2:]  # prev_choice, prev_reward (state in S and SS models)
                               ],
                    outputs_info=[p_right],
                    non_sequences=[n_subj, beta, persev])

            elif n_trials_back == 1:
                p_right, _ = theano.scan(  # shape: (n_trials-2, n_subj)
                    fn=p_from_Q_1,
                    sequences=[Qs,
                               choices[1:-1], rewards[1:-1],
                               choices[2:], rewards[2:]
                               ],
                    outputs_info=[p_right],
                    non_sequences=[n_subj, beta, persev])

            elif n_trials_back == 2:
                p_right, _ = theano.scan(  # shape: (n_trials-2, n_subj)
                    fn=p_from_Q_2,
                    sequences=[Qs,
                               choices[1:-1], rewards[1:-1],
                               choices[2:], rewards[2:]
                               ],
                    outputs_info=[p_right],
                    non_sequences=[n_subj, beta, persev])

            elif n_trials_back == 3:
                p_right, _ = theano.scan(  # shape: (n_trials-2, n_subj)
                    fn=p_from_Q_3,
                    sequences=[Qs,
                               choices[0:-2], rewards[0:-2],
                               choices[1:-1], rewards[1:-1],
                               choices[2:], rewards[2:]
                               ],
                    outputs_info=[p_right],
                    non_sequences=[n_subj, beta, persev])

        elif 'B' in model_name:

            scaled_persev_bonus = persev_bonus * persev
            lik_cor, lik_inc = get_likelihoods(rewards, choices, p_reward, p_noisy)

            # Get posterior & calculate probability of subsequent trial
            p_r = 0.5 * T.ones(n_subj, dtype='float32')
            [p_r, p_right, p_right0], _ = theano.scan(fn=post_from_lik,  # shape (n_trials, n_subj); starts predicting at trial 1!
                                            sequences=[lik_cor, lik_inc, scaled_persev_bonus],
                                            outputs_info=[p_r, None, None],
                                            non_sequences=[p_switch, beta])

            if 'b' in model_name:
                p_right = p_right[2:]  # predict from trial 3 onward, not trial 1 (consistent with RL / strat models)
            else:
                p_right = p_r[2:]  # use pure probability, without pushing through softmax (but after adding persev)

        # Use Bernoulli to sample responses
        if n_trials_back < 3:
            model_choices = pm.Bernoulli('model_choices', p=p_right[:-1], observed=choices[3:])  # predict from trial 3 on; discard last p_right because there is no trial to predict after the last value update
        else:
            model_choices = pm.Bernoulli('model_choices', p=p_right[:-1], observed=choices[4:])  # predict from trial 4 on; discard last p_right because there is no trial to predict after the last value update

        # Calculate NLL
        trialwise_LLs = pm.Deterministic('trialwise_LLs', T.log(p_right[:-1] * choices[3:] + (1 - p_right[:-1]) * (1 - choices[3:])))
        # subjwise_LLs = pm.Deterministic('subjwise_LLs', T.sum(trialwise_LLs, axis=0))

        # theano.printing.Print('beta')(beta)
        # theano.printing.Print('persev')(persev)
        # theano.printing.Print('p_switch')(p_switch)
        # theano.printing.Print('p_reward')(p_reward)
        # # theano.printing.Print('alpha')(alpha)
        # # theano.printing.Print('nalpha')(nalpha)
        # # theano.printing.Print('calpha')(calpha)
        # # theano.printing.Print('cnalpha')(cnalpha)
        # # theano.printing.Print('Qs')(Qs)
        # theano.printing.Print('choices')(choices)
        # theano.printing.Print('rewards')(rewards)
        # theano.printing.Print('p_right')(p_right)
        # theano.printing.Print('trialwise_LLs')(trialwise_LLs)
        # theano.printing.Print('T.sum(trialwise_LLs, axis=0)')(T.sum(trialwise_LLs, axis=0))
        # theano.printing.Print('T.sum(trialwise_LLs)')(T.sum(trialwise_LLs))

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


def fit_model_and_save(model, n_params, n_subj, n_trials, sIDs, slope_variable,
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
        # nll = opt_result['fun']  # this is approximately right, but does not agree with hand-calculated trialwise LLs
        nll = -np.sum(map['trialwise_LLs'])
        bic = np.log(n_trials * n_subj) * n_params + 2 * nll  # n_params incorporates all subj
        aic = 2 * n_params + 2 * nll
        print("NLL: {0}\nBIC: {1}\nAIC: {2}".format(nll, bic, aic))

    # Save results
    if fit_mcmc:
        print('Saving trace, model, summary, WAIC, and sIDs to {0}{1}\n'.format(save_dir, save_id))
        with open(save_dir + save_id + '_mcmc.pickle', 'wb') as handle:
            pickle.dump({'trace': trace, 'model': model, 'summary': model_summary, 'WAIC': waic.WAIC, 'slope_variable': slope_variable, 'sIDs': list(sIDs)},
                        handle, protocol=pickle.HIGHEST_PROTOCOL)
        pm.traceplot(trace)
        plt.savefig(save_dir + save_id + 'plot.png')
        return 0, 0, waic

    if fit_map:
        print('Saving map estimate, nll, bic, aic, sIDs to {0}{1}\n'.format(save_dir, save_id))
        # with open(save_dir + save_id + '_map.pickle', 'wb') as handle:  # TODO comment back in!
        with open(save_dir + 'gen_rec/' + save_id + '_map.pickle', 'wb') as handle:
            pickle.dump({'map': map, 'nll': nll, 'bic': bic, 'aic': aic, 'slope_variable': slope_variable, 'sIDs': list(sIDs)},
                        handle, protocol=pickle.HIGHEST_PROTOCOL)
        return nll, bic, aic


def get_slope_variables(model_name, kids_and_teens_only, adults_only):

    fit_slopes = any([i in model_name for i in 'lyouqtwv'])
    if fit_slopes and kids_and_teens_only:
        slope_variables = ['age_z', 'PDS_z', 'T1_log_z']
    elif fit_slopes and adults_only:
        slope_variables = ['T1_log_z']
    elif fit_slopes:
        raise ValueError("Fit slopes separately for children and adults! Set kids_and_teens_only=True or adults_only=True.")
    else:
        slope_variables = ['age_z']  # dummy; won't be used

    return slope_variables


# Determine the basics
contrast = 'quadratic'
n_groups = 1  # 'gender'  # 1
kids_and_teens_only = False
adults_only = False
if not run_on_cluster:
    fit_mcmc = False
    fit_map = True
    n_tune = 20
    n_samples = 20
    n_cores = 2
    n_chains = 1
else:
    fit_mcmc = True
    fit_map = False
    n_tune = 1000
    n_samples = 5000
    n_cores = 2
    n_chains = 2
target_accept = 0.8

if fit_mcmc:
    fit_individuals = False
else:
    fit_individuals = True


def replace_nans(data, n_trials):

    data = data[:n_trials]
    data[np.isnan(data)] = np.random.binomial(1, 0.5, np.sum(np.isnan(data)))

    return data


# Load behavioral data on which to run the model(s)
# Run all models
nll_bics = pd.DataFrame()
for model_name in model_names:

    n_subj, rewards, choices, group, n_groups, age = load_data(
        run_on_cluster, n_groups=n_groups, n_subj='all', kids_and_teens_only=kids_and_teens_only,  # n_groups can be 1, 2, 3 (for age groups) and 'gender" (for 2 gender groups)
        adults_only=adults_only, n_trials=120,
        fit_slopes=any([i in model_name for i in 'lyouqtwv' for model_name in model_names]))  # make sure I load the same data for every model...

    # n_subj, rewards, choices, group, n_groups, age = load_data(
    #     run_on_cluster, fitted_data_name='RL_simulations', n_groups=n_groups, n_subj='all', kids_and_teens_only=kids_and_teens_only,  # n_groups can be 1, 2, 3 (for age groups) and 'gender" (for 2 gender groups)
    #     adults_only=adults_only, n_trials=120,
    #     fit_slopes=any([i in model_name for i in 'lyouqtwv' for model_name in model_names]))  # make sure I load the same data for every model...

    # # Load mouse data
    # rewards_j = pd.read_csv('C:/Users/maria/MEGAsync/SLCN/PSMouseData/Juvi_Reward.csv').T.values  # row: sessions; cols: trials
    # choices_j = pd.read_csv('C:/Users/maria/MEGAsync/SLCN/PSMouseData/Juvi_Choice.csv').T.values
    # rewards_a = pd.read_csv('C:/Users/maria/MEGAsync/SLCN/PSMouseData/Adult_Reward.csv').T.values
    # choices_a = pd.read_csv('C:/Users/maria/MEGAsync/SLCN/PSMouseData/Adult_Choice.csv').T.values
    #
    # # Clean mouse data
    # n_trials_per_animal = np.sum(np.invert(np.isnan(rewards_j)), axis=0)
    # sns.distplot(n_trials_per_animal)
    # n_trials = np.round(np.percentile(n_trials_per_animal, 0.8)).astype('int')
    # rewards_j = replace_nans(rewards_j, n_trials).astype('int')
    # choices_j = replace_nans(choices_j, n_trials).astype('int')
    # rewards_a = replace_nans(rewards_a, n_trials).astype('int')
    # choices_a = replace_nans(choices_a, n_trials).astype('int')
    #
    # # Combine juvenile and adult data
    # rewards = np.hstack([rewards_j, rewards_a])
    # choices = np.hstack([choices_j, choices_a])
    #
    # n_subj = np.shape(rewards)[1]
    # assert np.shape(rewards) == np.shape(choices)
    # group = np.zeros(n_subj)
    # n_groups = len(np.unique(group))
    # age = pd.DataFrame({'age_z': np.zeros(n_subj), 'PDS_z': np.zeros(n_subj), 'T_z': np.zeros(n_subj)})
    # sID_j = pd.read_csv('C:/Users/maria/MEGAsync/SLCN/PSMouseData/Juvi_AnimalID.csv').T.values.flatten()
    # sID_a = pd.read_csv('C:/Users/maria/MEGAsync/SLCN/PSMouseData/Adult_AnimalID.csv').T.values.flatten()
    # age['sID'] = np.concatenate([sID_j, sID_a])

    # slope_variables = get_slope_variables(model_name, kids_and_teens_only, adults_only)
    slope_variables = ['age_z']  # ['PDS_z', 'T1_log_z']
    for slope_variable in slope_variables:

        # Create model
        model, n_params, n_trials, save_dir, save_id = create_model(
            choices=choices, rewards=rewards, group=group, age=age, n_groups=n_groups,
            model_name=model_name, slope_variable=slope_variable, contrast=contrast, fit_individuals=fit_individuals,
            n_subj=n_subj, n_trials='all', verbose=False, print_logps=False)

        # Fit parameters, calculate model fits, make plots, and save everything
        nll, bic, aic = fit_model_and_save(model, n_params, n_subj, n_trials, age['sID'], slope_variable,
                                           save_dir, save_id,
                                           fit_mcmc=fit_mcmc, fit_map=fit_map,
                                           n_samples=n_samples, n_tune=n_tune, n_chains=n_chains, n_cores=n_cores)

        nll_bics = nll_bics.append([[model_name, slope_variable, n_params, nll, bic, aic]])
        nll_bics.to_csv(save_dir + '/nll_bics_temp.csv', index=False)

nll_bics.columns = ['model_name', 'slope_variable', 'n_params', 'nll', 'bic', 'aic']
nll_bics.to_csv(save_dir + '/nll_bics.csv', index=False)
