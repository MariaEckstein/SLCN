import numpy as np
import pymc3 as pm
import theano.tensor as T


def get_slope(param_name, sd_testval, slope_testval, n_groups, sd=10):

    if n_groups > 1:
        param_slope_mu = pm.Normal(param_name + '_slope_mu', mu=0, sd=sd, shape=1, testval=slope_testval * T.ones(1, dtype='int32'))
        param_slope_sd = pm.HalfNormal(param_name + '_slope_sd', sd=sd, shape=1, testval=sd_testval * T.ones(1, dtype='int32'))
    else:
        param_slope_mu, param_slope_sd = 0, sd

    return pm.Normal(param_name + '_slope', mu=param_slope_mu, sd=param_slope_sd, shape=n_groups, testval=slope_testval * T.ones(n_groups, dtype='int32'))


def get_sd(param_name, sd_testval, n_groups, sd=10):

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
        param_sd = get_sd(param_name, sd_testval, n_groups)

        if np.any([contrast == c for c in ['linear', 'quadratic', 'cubic', '4th']]):
            param_slope = get_slope(param_name, sd_testval, slope_testval, n_groups)
            print("Adding slope for {}.".format(param_name))
        else:
            param_slope = T.zeros(n_groups)

        if np.any([contrast == c for c in ['quadratic', 'cubic', '4th']]):
            param_slope2 = get_slope(param_name + '_2', sd_testval, slope_testval, n_groups)
            print("Adding quadratic slope for {}.".format(param_name))
        else:
            param_slope2 = T.zeros(n_groups)

        if np.any([contrast == c for c in ['cubic', '4th']]):
            param_slope3 = get_slope(param_name + '_3', sd_testval, slope_testval, n_groups)
            print("Adding cubic slope for {}.".format(param_name))
        else:
            param_slope3 = T.zeros(n_groups)

        if np.any([contrast == c for c in ['4th']]):
            param_slope4 = get_slope(param_name + '_4', sd_testval, slope_testval, n_groups)
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
