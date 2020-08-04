run_on_cluster = False
save_dir_appx = 'mice/'  # ''

# GET LIST OF MODELS TO RUN
model_names = [
    'RLabnp2', 'Bbspr'
    # 'RLabnp2dlyqt', 'Bbsprywtv'
    # 'RLab', 'RLabd', 'RLabcd', 'RLabcpd', 'RLabcpnd', 'RLabnp2', 'RLabnp2d', 'RLabcpnxd',
    # 'Bbspr', 'Bbpr', 'Bbp', 'Bb', 'B',
    # 'WSLSy', 'WSLSSy', 'WSLSdfy', 'WSLSSdfy',
]

# Legend for letters -> parameters
## All models
# 'b' -> beta; 'y' -> slope
# 'p' -> choice perseverance / sticky choice; 't' -> slope
## RL models
# 'a' -> alpha; 'l' -> slope
# 'c' -> counterfactual alpha; 'o' -> slope
# 'n' -> negative alpha; 'q' -> slope
# 'x' -> counterfactual negative alpha; 'u' -> slope
# 'd' -> left-bias; 'f' -> slope
# 'e' -> counterfactual update for negative outcomes; 'g' -> slope
# 'm' -> ???; 'z' -> slope
## Bayesian models
# 's' -> p_switch; 'w' -> slope
# 'r' -> p_reward; 'v' -> slope

run_on = 'mice'  # 'humans', 'mice'

print("Getting ready to run {} {} models: {}".format(len(model_names), run_on, model_names))

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import pymc3 as pm
import theano
import theano.tensor as T
from PSModelFunctions import create_parameter, get_slope_variables
from PSModelFunctions2 import get_n_params, update_Q, get_likelihoods, post_from_lik, p_from_Q, p_from_prev_WSLS, p_from_prev_WSLSS
from PSModelFunctions3 import load_data, load_mouse_data_for_modeling, get_save_dir_and_save_id, print_logp_info

floatX = 'float32'
theano.config.floatX = 'float32'
theano.config.warn_float64 = 'warn'


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
        persev_bonus = 2 * choices - 1  # -1 for choice==0 (left) and +1 for choice==1 (right)
        persev_bonus = theano.shared(np.asarray(persev_bonus, dtype='int16'))

    # Transform everything into theano.shared variables
    rewards = theano.shared(np.asarray(rewards, dtype='int16'))
    choices = theano.shared(np.asarray(choices, dtype='int16'))
    group = theano.shared(np.asarray(group, dtype='int16'))
    slope_variable = theano.shared(np.asarray(age[slope_variable], dtype='float32'))

    # Get n_trials_back, n_params, and file_name_suff
    n_params = get_n_params(model_name, n_subj, n_groups, contrast=contrast)
    file_name_suff = model_name + ''
    save_dir, save_id = get_save_dir_and_save_id(run_on_cluster, file_name_suff, fitted_data_name, n_samples)
    save_dir += save_dir_appx
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(save_dir + 'plots/')

    print("Working on model '{}', which has {} free parameters. Save_dir: {}".format(model_name, n_params, save_dir))
    print("Compiling models for {} {} with {} samples and {} tuning steps...\n".format(n_subj, fitted_data_name, n_samples, n_tune))

    with pm.Model() as model:
        if not fit_individuals:  # fit_MCMC == True

            # RL, Bayes, and WSLS
            if ('b' in model_name) or ('WSLS' in model_name):
                beta = create_parameter('beta', 'Gamma', 'y' in model_name, n_groups, group, n_subj, upper, slope_variable, contrast)
            else:
                beta = pm.Gamma('beta', alpha=1, beta=1, shape=n_subj)  # won't be used - necessary for sampling
                print("This model does not have beta.")

            if 'p' in model_name:
                persev = create_parameter('persev', 'Normal', 't' in model_name, n_groups, group, n_subj, upper, slope_variable, contrast)
            else:
                persev = pm.Deterministic('persev', T.zeros(n_subj, dtype='float32'))
                print("Setting persev = 0.")

            if 'd' in model_name:
                if 'f' in model_name:
                    bias_intercept = pm.Uniform('bias_intercept', lower=-1, upper=1, shape=n_groups, testval=0.1 * T.ones(n_groups, dtype='int32'))
                    bias_slope = pm.Uniform('bias_slope', lower=-1, upper=1, shape=n_groups, testval=-0.1 * T.ones(n_groups, dtype='int32'))
                    if contrast == 'quadratic':
                        bias_slope2 = pm.Uniform('bias_slope2', lower=-1, upper=1, shape=n_groups, testval=-0.1 * T.ones(n_groups, dtype='int32'))
                    else:
                        bias_slope2 = T.zeros(n_groups, dtype='int16')
                    bias = pm.Deterministic('bias', bias_intercept[group] + bias_slope[group] * slope_variable + bias_slope2[group] * T.sqr(slope_variable))
                    print("Drawing slope, intercept, and noise for bias.")
                else:
                    bias = pm.Uniform('bias', lower=-1, upper=1, shape=n_subj, testval=0.1 * T.ones(n_subj, dtype='float32'))
                print("Adding free parameter bias.")
            else:
                bias = pm.Deterministic('bias', T.zeros(n_subj, dtype='float32'))

            if 'RL' in model_name:
                if 'a' in model_name:
                    alpha = create_parameter('alpha', 'Beta', 'l' in model_name, n_groups, group, n_subj, upper, slope_variable, contrast)
                else:
                    alpha = pm.Deterministic('alpha', T.ones(n_subj, dtype='float32'))
                    print("Setting alpha = 1")

                if 'n' in model_name:
                    nalpha = create_parameter('nalpha', 'Beta', 'q' in model_name, n_groups, group, n_subj, upper, slope_variable, contrast)
                else:
                    nalpha = pm.Deterministic('nalpha', 1 * alpha)
                    print("Setting nalpha = alpha.")

                if 'c' in model_name:
                    calpha = create_parameter('calpha', 'Beta', 'o' in model_name, n_groups, group, n_subj, upper, slope_variable, contrast)
                elif '2' in model_name:
                    calpha = alpha.copy()
                    print("Setting calpha = alpha.")
                else:
                    calpha = pm.Deterministic('calpha', T.zeros(n_subj, dtype='float32'))
                    print("Setting calpha = 0.")

                if 'x' in model_name:
                    cnalpha = create_parameter('cnalpha', 'Beta', 'u' in model_name, n_groups, group, n_subj, upper, slope_variable, contrast)
                elif '2' in model_name:
                    cnalpha = nalpha.copy()
                    print("Setting cnalpha = nalpha.")
                else:
                    cnalpha = pm.Deterministic('cnalpha', T.zeros(n_subj, dtype='float32'))

                if (('x' in model_name) or ('2' in model_name)) and ('e' in model_name):
                    cnalpha_rew = create_parameter('cnalpha_rew', 'Beta', 'g' in model_name, n_groups, group, n_subj, upper, slope_variable, contrast)
                    print("Adding free parameter cnalpha_rew.")
                else:
                    cnalpha_rew = pm.Deterministic('cnalpha_rew', T.ones(n_subj, dtype='float32'))
                    print("Setting cnalpha_rew = 1.")

            elif 'B' in model_name:

                p_noisy = 1e-5 * T.as_tensor_variable(1)

                if 's' in model_name:
                    p_switch = create_parameter('p_switch', 'Beta', 'w' in model_name, n_groups, group, n_subj, upper, slope_variable, contrast)
                else:
                    p_switch = pm.Deterministic('p_switch', 0.05081582 * T.ones(n_subj, dtype='float32'))
                    print("Setting p_switch to 0.0508...")

                if 'r' in model_name:
                    p_reward = create_parameter('p_reward', 'Beta', 'v' in model_name, n_groups, group, n_subj, upper, slope_variable, contrast)
                else:
                    p_reward = pm.Deterministic('p_reward', 0.75 * T.ones(n_subj, dtype='float32'))
                    print("Setting p_reward to 0.75.")

        else:  # if fit_individuals == True:

            beta = pm.Uniform('beta', lower=0, upper=15, shape=n_subj, testval=2 * T.ones(n_subj, dtype='float32'))
            print("Adding free parameter beta (Uniform 0 15).")

            if 'p' in model_name:  # all models
                persev = pm.Uniform('persev', lower=-1, upper=1, shape=n_subj, testval=0 * T.ones(n_subj, dtype='float32'))
                print("Adding free parameter persev (Uniform -1 1).")
            else:
                persev = pm.Deterministic('persev', T.zeros(n_subj, dtype='float32'))

            if 'd' in model_name:
                bias = pm.Uniform('bias', lower=-1, upper=1, shape=n_subj, testval=0.1 * T.ones(n_subj, dtype='float32'))
                print("Adding free parameter bias (Uniform -1 1).")
            else:
                bias = pm.Deterministic('bias', T.zeros(n_subj, dtype='float32'))

            if 'RL' in model_name:  # only RL models

                if 'a' in model_name:
                    alpha = pm.Beta('alpha', alpha=1, beta=1, shape=n_subj, testval=0.5 * T.ones(n_subj, dtype='float32'))
                    print("Adding free parameter alpha (Beta 1 1).")
                else:
                    alpha = pm.Deterministic('alpha', T.ones(n_subj, dtype='float32'))
                    print("Setting alpha = 1.")

                if 'n' in model_name:
                    nalpha = pm.Beta('nalpha', alpha=1, beta=1, shape=n_subj, testval=0.5 * T.ones(n_subj, dtype='float32'))
                    print("Adding free parameter nalpha (Beta 1 1).")
                else:
                    nalpha = pm.Deterministic('nalpha', 1 * alpha)
                    print("Setting nalpha = alpha.")

                if 'c' in model_name:
                    calpha = pm.Beta('calpha', alpha=1, beta=1, shape=n_subj, testval=0.5 * T.ones(n_subj, dtype='float32'))
                    print("Adding free parameter calpha (Beta 1 1).")

                elif '2' in model_name:
                    calpha = pm.Deterministic('calpha', 1 * alpha)
                    print("Setting calpha = alpha")

                else:
                    calpha = 0
                    print("Setting calpha = 0.")

                if 'x' in model_name:
                    cnalpha = pm.Beta('cnalpha', alpha=1, beta=1, shape=n_subj, testval=0.5 * T.ones(n_subj, dtype='float32'))
                    print("Adding free parameter cnalpha (Beta 1 1).")

                elif '2' in model_name:
                    cnalpha = pm.Deterministic('cnalpha', 1 * nalpha)
                    print("Setting cnalpha = nalpha")

                else:
                    cnalpha = 0
                    print("Setting cnalpha = 0.")

                if (('x' in model_name) or ('2' in model_name)) and ('e' in model_name):
                    cnalpha_rew = pm.Beta('cnalpha_rew', alpha=1, beta=1, shape=n_subj, testval=0.1 * T.ones(n_subj, dtype='float32'))
                    print("Adding free parameter cnalpha_rew (Beta 1 1).")
                else:
                    cnalpha_rew = pm.Deterministic('cnalpha_rew', T.ones(n_subj, dtype='float32'))
                    print("Setting cnalpha_rew = 1.")

            elif 'B' in model_name:  # only BF models
                p_noisy = pm.Deterministic('p_noisy', 1e-5 * T.ones(n_subj, dtype='float32'))
                if 's' in model_name:
                    p_switch = pm.Beta('p_switch', alpha=1, beta=1, shape=n_subj, testval=0.0508 * T.ones(n_subj, dtype='float32'))
                    print("Adding free parameter p_switch (Beta 1 1).")
                else:
                    p_switch = pm.Deterministic('p_switch', 0.0508 * T.ones(n_subj))  # checked on 2019-06-03 in R: `mean(all_files$switch_trial)`
                    print("Setting p_switch = 0.05081582.")

                if 'r' in model_name:
                    p_reward = pm.Beta('p_reward', alpha=1, beta=1, shape=n_subj, testval=0.75 * T.ones(n_subj, dtype='float32'))
                    print("Adding free parmeter p_reward (Beta 1 1).")
                else:
                    p_reward = pm.Deterministic('p_reward', 0.75 * T.ones(n_subj))  # 0.75 because p_reward is the prob. of getting reward if choice is correct
                    print("Setting p_reward = 0.75.")

        # Initialize Q-values
        if 'RL' in model_name:
            if 'ab' in model_name:  # letter models
                Qs = 0.5 * T.ones((n_subj, 2), dtype='float32')
            _ = T.ones(n_subj, dtype='float32')

            # Calculate Q-values for all trials (RL models only)
            [Qs, _], _ = theano.scan(  # shape: (n_trials-2, n_subj, prev_choice); starts predicting at trial 3!
                fn=update_Q,
                sequences=[
                    # choices[0:-2], rewards[0:-2],
                    # choices[1:-1], rewards[1:-1],
                    choices[2:], rewards[2:]],
                outputs_info=[Qs, _],
                non_sequences=[alpha, nalpha, calpha, cnalpha, cnalpha_rew, n_subj])

            # Initialize p_right for first trial
            p_right = 0.5 * T.ones(n_subj, dtype='float32')  # shape: (n_subj)

            # Translate Q-values into probabilities for all trials
            p_right, _ = theano.scan(  # shape: (n_trials-2, n_subj)
                fn=p_from_Q,
                sequences=[Qs,
                           # choices[1:-1], rewards[1:-1],
                           choices[2:], #rewards[2:]
                           ],
                outputs_info=[p_right],
                non_sequences=[n_subj, beta, persev, bias])

        elif 'WSLS' in model_name:

            p_right = 0.5 * T.ones(n_subj, dtype='float32')  # shape: (n_subj)
            p_right, _ = theano.scan(
                fn=p_from_prev_WSLS,
                sequences=[choices[2:], rewards[2:],],
                outputs_info=[p_right],
                non_sequences=[beta, bias]
            )

        elif 'WSLSS' in model_name:

            p_right = 0.5 * T.ones(n_subj, dtype='float32')  # shape: (n_subj)
            p_right, _ = theano.scan(
                fn=p_from_prev_WSLSS,
                sequences=[choices[1:-1], rewards[1:-1], choices[2:], rewards[2:],],
                outputs_info=[p_right],
                non_sequences=[beta, bias]
            )

        elif 'B' in model_name:

            scaled_persev_bonus = persev_bonus * persev
            lik_cor, lik_inc = get_likelihoods(rewards, choices, p_reward, p_noisy)

            # Get posterior & calculate probability of subsequent trial
            p_r = 0.5 * T.ones(n_subj, dtype='float32')
            [p_r, p_right, p_right0], _ = theano.scan(fn=post_from_lik,  # shape (n_trials, n_subj); starts predicting at trial 1!
                                            sequences=[lik_cor, lik_inc, scaled_persev_bonus],
                                            outputs_info=[p_r, None, None],
                                            non_sequences=[p_switch, beta, bias])

            if 'b' in model_name:
                p_right = p_right[2:]  # predict from trial 3 onward, not trial 1 (consistent with RL / strat models)
            else:
                p_right = p_r[2:]  # use pure probability, without pushing through softmax (but after adding persev)

        # Use Bernoulli to sample responses
        model_choices = pm.Bernoulli('model_choices', p=p_right[:-1], observed=choices[3:])  # predict from trial 3 on; discard last p_right because there is no trial to predict after the last value update

        # Calculate NLL
        trialwise_LLs = pm.Deterministic('trialwise_LLs', T.log(p_right[:-1] * choices[3:] + (1 - p_right[:-1]) * (1 - choices[3:])))
        # subjwise_LLs = pm.Deterministic('subjwise_LLs', T.sum(trialwise_LLs, axis=0))

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
        print("MCMC estimates: {0}\nWAIC: {1}".format(model_summary, waic.waic))

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
            pickle.dump({'trace': trace, 'model': model, 'summary': model_summary, 'WAIC': waic.waic, 'slope_variable': slope_variable, 'sIDs': list(sIDs)},
                        handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Saved everything!")
        # pm.traceplot(trace)
        # plt.savefig(save_dir + save_id + 'plot.png')
        return 0, 0, waic

    if fit_map:
        print('Saving map estimate, nll, bic, aic, sIDs to {0}{1}\n'.format(save_dir, save_id))
        with open(save_dir + save_id + '_map.pickle', 'wb') as handle:
        # with open(save_dir + 'gen_rec/' + save_id + '_map.pickle', 'wb') as handle:
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
contrast = 'linear'
n_groups = 1  # 'gender'  # 1
kids_and_teens_only = False
adults_only = False
if not run_on_cluster:
    fit_mcmc = False  # False
    fit_individuals = True  # True
    fit_map = True
    n_tune = 20  # 100  # 20
    n_samples = 20  # 300  # 20
    n_cores = 2
    n_chains = 1
else:
    fit_mcmc = True
    fit_individuals = False
    fit_map = False
    n_tune = 1000
    n_samples = 5000
    n_cores = 1  # 2
    n_chains = 2
target_accept = 0.8


# Load behavioral data on which to run the model(s)
# Run all models
nll_bics = pd.DataFrame()
for model_name in model_names:

    if run_on == 'humans':

        # Load human data
        n_subj, rewards, choices, group, n_groups, age = load_data(
            run_on_cluster, n_groups=n_groups, n_subj='all', kids_and_teens_only=kids_and_teens_only,  # n_groups can be 1, 2, 3 (for age groups) and 'gender" (for 2 gender groups)
            adults_only=adults_only, n_trials=120,
            fit_slopes=any([i in model_name for i in 'lyouqtwv' for model_name in model_names]))  # make sure I load the same data for every model...

        # n_subj, rewards, choices, group, n_groups, age = load_data(
        #     run_on_cluster, fitted_data_name='BF_simulations', n_groups=n_groups, n_subj='all', kids_and_teens_only=kids_and_teens_only,  # n_groups can be 1, 2, 3 (for age groups) and 'gender" (for 2 gender groups)
        #     adults_only=adults_only, n_trials=120,
        #     fit_slopes=any([i in model_name for i in 'lyouqtwv' for model_name in model_names]))  # make sure I load the same data for every model...

    elif run_on == 'mice':

        # Load mouse data
        n_subj, rewards, choices, group, n_groups, age = load_mouse_data_for_modeling(
            'mice',  # 'simulations'
            first_session_only=False,
            fit_sessions_individually=True,
            # simulation_name='simulated_mice_WSLSSd_nagents10.csv',
            )

    # Saving as csv
    ages_dir = 'C:/Users/maria/MEGAsync/SLCN/PSMouseData/age.csv'
    print("Saving ages to " + ages_dir)
    age.to_csv(ages_dir, index=False)

    slope_variables = ['session']  # For mice: Include sloep letters to use the slope variable; will be ignore otherwise # ['age_z', 'PDS_z', 'meanT_log_z']  # get_slope_variables(model_name, kids_and_teens_only, adults_only)
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
