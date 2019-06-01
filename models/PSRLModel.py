import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import pymc3 as pm
import seaborn as sns
import theano
import theano.tensor as T

from shared_modeling_simulation import get_strat_Qs, get_WSLS_Qs, update_Q_0, update_Q_1, update_Q_2, p_from_Q_0, p_from_Q_1, p_from_Q_2
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
                 model_name='ab',  # ab, abc, abn, abcn, abcnnc, abcnS, abcnSS, WSLS, strat, etc. etc.
                 verbose=False, print_logps=False,
                 fitted_data_name='humans',  # 'humans', 'simulations'
                 n_groups=3, fit_individuals=True,
                 upper=1000,
                 ):

    # Set stuff
    file_name_suff = model_name + ''

    # Get n_trials_back for each model
    if 'SSS' in model_name:  # abSSS, etc.
        n_trials_back = 3
    elif 'SS' in model_name or 'strat' in model_name:  # strat, abSS, etc.
        n_trials_back = 2
    elif 'S' in model_name:  # WSLS and abS, abcS, etc.
        n_trials_back = 1
    else:  # ab, abc, etc.
        n_trials_back = 0

    # Get n_params for each model
    if 'abcnnc' in model_name:  # abcnnc, abcnncS, abcnncSS, etc.
        n_params = 5
    elif 'abcn' in model_name:
        n_params = 4
    elif 'abc' in model_name or 'abn' in model_name:
        n_params = 3
    elif 'WSLS' in model_name or 'strat' in model_name or 'ab' in model_name:
        n_params = 1  # beta
    print("Working on model '{0}', which has {1} free parameters.".format(model_name, n_params))

    # Load to-be-fitted data from n_subj subjects with n_trials trials
    if n_trials == 'all':
        n_trials = len(choices)
    else:
        choices = choices[:n_trials]
        rewards = rewards[:n_trials]

    # Create choices_both (first trial is persev_bonus for second trial = where Qs starts)
    choices_both = np.array([1 - choices, choices])  # left, right
    choices_both = np.transpose(choices_both, (1, 2, 0))[:-1]  # make same shape as Qs (n_trials-1, n_subj, 2); remove last trial because it's not needed

    # Get fixed Q-values for WSLS and strat
    if 'strat' in model_name:
        # Strat model: "stay unless failed to receive reward twice in a row for the same action."
        Qs = get_strat_Qs(n_trials, n_subj)
        Qs = theano.shared(np.asarray(Qs, dtype='float32'))

    elif 'WSLS' in model_name:
        # WSLS model: "stay if you won; switch if you lost"
        Qs = get_WSLS_Qs(n_trials, n_subj)
        Qs = theano.shared(np.asarray(Qs, dtype='float32'))

    # Transform everything into theano.shared variables
    rewards = theano.shared(np.asarray(rewards, dtype='int32'))
    choices = theano.shared(np.asarray(choices, dtype='int32'))
    choices_both = theano.shared(np.asarray(choices_both, dtype='int32'))
    group = theano.shared(np.asarray(group, dtype='int32'))

    # Prepare things for saving
    save_dir, save_id = get_save_dir_and_save_id(run_on_cluster, file_name_suff, fitted_data_name, n_samples)

    # Fit models
    print("Compiling models for {0} with {1} samples and {2} tuning steps...\n".format(fitted_data_name, n_samples, n_tune))

    with pm.Model() as model:

        if not fit_individuals:
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

            # cnalpha_sc_a_a = pm.Uniform('cnalpha_sc_a_a', lower=0, upper=upper)
            # cnalpha_sc_a_b = pm.Uniform('cnalpha_sc_a_b', lower=0, upper=upper)
            # cnalpha_sc_b_a = pm.Uniform('cnalpha_sc_b_a', lower=0, upper=upper)
            # cnalpha_sc_b_b = pm.Uniform('cnalpha_sc_b_b', lower=0, upper=upper)
            # cnalpha_sc_a = pm.Gamma('cnalpha_sc_a', alpha=cnalpha_sc_a_a, beta=cnalpha_sc_a_b, shape=n_groups)
            # cnalpha_sc_b = pm.Gamma('cnalpha_sc_b', alpha=cnalpha_sc_b_a, beta=cnalpha_sc_b_b, shape=n_groups)
            # cnalpha_sc = pm.Beta('cnalpha_sc', alpha=cnalpha_sc_a[group], beta=cnalpha_sc_b[group], shape=n_subj)
            cnalpha_sc = pm.Deterministic('cnalpha_sc', calpha_sc.copy())

            # Get parameter means and variances
            alpha_mu = pm.Deterministic('alpha_mu', 1 / (1 + alpha_b / alpha_a))
            alpha_var = pm.Deterministic('alpha_var', (alpha_a * alpha_b) / (np.square(alpha_a + alpha_b) * (alpha_a + alpha_b + 1)))
            beta_mu = pm.Deterministic('beta_mu', beta_a / beta_b)
            beta_var = pm.Deterministic('beta_var', beta_a / np.square(beta_b))
            nalpha_mu = pm.Deterministic('nalpha_mu', 1 / (1 + nalpha_b / nalpha_a))
            nalpha_var = pm.Deterministic('nalpha_var', (nalpha_a * nalpha_b) / (np.square(nalpha_a + nalpha_b) * (nalpha_a + nalpha_b + 1)))
            calpha_sc_mu = pm.Deterministic('calpha_sc_mu', 1 / (1 + calpha_sc_b / calpha_sc_a))
            calpha_sc_var = pm.Deterministic('calpha_sc_var', (calpha_sc_a * calpha_sc_b) / (np.square(calpha_sc_a + calpha_sc_b) * (calpha_sc_a + calpha_sc_b + 1)))
            # cnalpha_sc_mu = pm.Deterministic('cnalpha_sc_mu', 1 / (1 + cnalpha_sc_b / cnalpha_sc_a))
            # cnalpha_sc_var = pm.Deterministic('cnalpha_sc_var', (cnalpha_sc_a * cnalpha_sc_b) / (np.square(cnalpha_sc_a + cnalpha_sc_b) * (cnalpha_sc_a + cnalpha_sc_b + 1)))

            # Individual parameters (with group-level priors)
            alpha = pm.Beta('alpha', alpha=alpha_a[group], beta=alpha_b[group], shape=n_subj, testval=0.5 * T.ones(n_subj))
            beta = pm.Gamma('beta', alpha=beta_a[group], beta=beta_b[group], shape=n_subj, testval=T.ones(n_subj))
            persev = 0
            persev_bonus = persev * choices_both

            nalpha = pm.Beta('nalpha', alpha=nalpha_a[group], beta=nalpha_b[group], shape=n_subj, testval=0.5 * T.ones(n_subj))
            calpha = pm.Deterministic('calpha', alpha * calpha_sc)
            cnalpha = pm.Deterministic('cnalpha', nalpha * cnalpha_sc)

            # Group differences?
            alpha_mu_diff01 = pm.Deterministic('alpha_mu_diff01', alpha_mu[0] - alpha_mu[1])
            alpha_mu_diff02 = pm.Deterministic('alpha_mu_diff02', alpha_mu[0] - alpha_mu[2])
            alpha_mu_diff12 = pm.Deterministic('alpha_mu_diff12', alpha_mu[1] - alpha_mu[2])

            beta_mu_diff01 = pm.Deterministic('beta_mu_diff01', beta_mu[0] - beta_mu[1])
            beta_mu_diff02 = pm.Deterministic('beta_mu_diff02', beta_mu[0] - beta_mu[2])
            beta_mu_diff12 = pm.Deterministic('beta_mu_diff12', beta_mu[1] - beta_mu[2])

            nalpha_mu_diff01 = pm.Deterministic('nalpha_mu_diff01', nalpha_mu[0] - nalpha_mu[1])
            nalpha_mu_diff02 = pm.Deterministic('nalpha_mu_diff02', nalpha_mu[0] - nalpha_mu[2])
            nalpha_mu_diff12 = pm.Deterministic('nalpha_mu_diff12', nalpha_mu[1] - nalpha_mu[2])

            calpha_sc_mu_diff01 = pm.Deterministic('calpha_sc_mu_diff01', calpha_sc_mu[0] - calpha_sc_mu[1])
            calpha_sc_mu_diff02 = pm.Deterministic('calpha_sc_mu_diff02', calpha_sc_mu[0] - calpha_sc_mu[2])
            calpha_sc_mu_diff12 = pm.Deterministic('calpha_sc_mu_diff12', calpha_sc_mu[1] - calpha_sc_mu[2])

            # cnalpha_sc_mu_diff01 = pm.Deterministic('cnalpha_sc_mu_diff01', cnalpha_sc_mu[0] - cnalpha_sc_mu[1])
            # cnalpha_sc_mu_diff02 = pm.Deterministic('cnalpha_sc_mu_diff02', cnalpha_sc_mu[0] - cnalpha_sc_mu[2])
            # cnalpha_sc_mu_diff12 = pm.Deterministic('cnalpha_sc_mu_diff12', cnalpha_sc_mu[1] - cnalpha_sc_mu[2])

        else:  # if fit_individuals == True:
            beta = pm.Gamma('beta', alpha=1, beta=1, shape=n_subj)
            if 'a' in model_name:
                alpha = pm.Beta('alpha', alpha=1, beta=1, shape=n_subj)
            else:
                alpha = pm.Deterministic('alpha', T.ones(1, dtype='float32'))
            if 'n' in model_name:
                nalpha = pm.Beta('nalpha', alpha=1, beta=1, shape=n_subj)
            else:
                nalpha = pm.Deterministic('nalpha', 1 * alpha)
            if 'c' in model_name:
                calpha_sc = pm.Beta('calpha_sc', alpha=1, beta=1, shape=n_subj)
            else:
                calpha_sc = 0
            if 'nc' in model_name:
                cnalpha_sc = pm.Beta('cnalpha_sc', alpha=1, beta=1, shape=n_subj)
            elif 'c' in model_name:
                cnalpha_sc = pm.Deterministic('cnalpha_sc', 1 * calpha_sc)
            else:
                cnalpha_sc = 0
            calpha = pm.Deterministic('calpha', alpha * calpha_sc)
            cnalpha = pm.Deterministic('cnalpha', nalpha * cnalpha_sc)
            persev = 0
            persev_bonus = persev * choices_both

        # Initialize Q-values for a single trial
        if 'SS' in model_name:  # SS models
            if 'i' in model_name:
                Qs = get_strat_Qs(n_trials, n_subj)[0]
                Qs = theano.shared(np.array(Qs, dtype='float32'))  # initialize at strat Qs
            else:
                Qs = 0.5 * T.ones((n_subj, 2, 2, 2, 2, 2), dtype='float32')  # (n_subj, prev_prev_choice, prev_prev_reward, prev_choice, prev_reward, choice)
        elif 'S' in model_name and 'WSLS' not in model_name:  # S models
            if 'i' in model_name:
                Qs = get_WSLS_Qs(n_trials, n_subj)[0]
                Qs = theano.shared(np.array(Qs, dtype='float32'))  # initialize at WSLS Qs
            else:
                Qs = 0.5 * T.ones((n_subj, 2, 2, 2), dtype='float32')  # (n_subj, prev_choice, prev_reward, choice)
        elif 'ab' in model_name:  # letter models
            Qs = 0.5 * T.ones((n_subj, 2), dtype='float32')
        _ = T.ones(n_subj, dtype='float32')

        # Calculate Q-values for all trials (not for WSLS and strat)
        if 'strat' not in model_name and 'WSLS' not in model_name:
            if n_trials_back == 0:
                [Qs, _], _ = theano.scan(  # shape: (n_trials-2, n_subj, prev_choice)
                    fn=update_Q_0,
                    sequences=[
                        choices[0:-2], rewards[0:-2],  # prev_prev_choice, prev_prev_reward: 1st trial til 3rd-to-last
                        choices[1:-1], rewards[1:-1],  # prev_choice, prev_reward: 2nd trial til 2nd-to-last
                        choices[2:], rewards[2:]],  # choice, reward: 3rd trial til last (for updating)
                    outputs_info=[Qs, _],
                    non_sequences=[alpha, nalpha, calpha, cnalpha, n_subj])

            elif n_trials_back == 1:
                [Qs, _], _ = theano.scan(  # shape: (n_trials-2, n_subj, prev_choice, prev_reward, choice)
                    fn=update_Q_1,
                    sequences=[
                        choices[0:-2], rewards[0:-2],  # prev_prev_choice, prev_prev_reward: 1st trial til 3rd-to-last
                        choices[1:-1], rewards[1:-1],  # prev_choice, prev_reward: 2nd trial til 2nd-to-last
                        choices[2:], rewards[2:]],  # choice, reward: 3rd trial til last (for updating)
                    outputs_info=[Qs, _],
                    non_sequences=[alpha, nalpha, calpha, cnalpha, n_subj])

            elif n_trials_back == 2:
                [Qs, _], _ = theano.scan(  # shape: (n_trials-2, n_subj, prev_prev_choice, prev_prev_reward, prev_choice, prev_reward, choice)
                    fn=update_Q_2,
                    sequences=[
                        choices[0:-2], rewards[0:-2],  # prev_prev_choice, prev_prev_reward: 1st trial til 3rd-to-last
                        choices[1:-1], rewards[1:-1],  # prev_choice, prev_reward: 2nd trial til 2nd-to-last
                        choices[2:], rewards[2:]],  # choice, reward: 3rd trial til last (for updating)
                    outputs_info=[Qs, _],
                    non_sequences=[alpha, nalpha, calpha, cnalpha, n_subj])

        # Initialize p_right for a single trial
        p_right = 0.5 * T.ones(n_subj, dtype='float32')  # shape: (n_subj)

        # Translate Q-values into probabilities for all trials
        if n_trials_back == 0:
            p_right, _ = theano.scan(  # shape: (n_trials-2, n_subj)
                fn=p_from_Q_0,
                sequences=[Qs, persev_bonus[1:],
                           choices[0:-2], rewards[0:-2],
                           choices[1:-1], rewards[1:-1]
                           ],
                outputs_info=[p_right],
                non_sequences=[n_subj, beta])

        elif n_trials_back == 1:
            p_right, _ = theano.scan(  # shape: (n_trials-2, n_subj)
                fn=p_from_Q_1,
                sequences=[Qs, persev_bonus[1:],
                           choices[0:-2], rewards[0:-2],
                           choices[1:-1], rewards[1:-1]
                           ],
                outputs_info=[p_right],
                non_sequences=[n_subj, beta])

        elif n_trials_back == 2:
            p_right, _ = theano.scan(  # shape: (n_trials-2, n_subj)
                fn=p_from_Q_2,
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
            theano.printing.Print('p_right')(p_right)

    return model, n_params, n_trials, save_dir, save_id


def get_results(model, n_params, n_subj, n_trials,
                fit_mcmc, fit_map,
                n_samples, n_tune, n_chains, n_cores, target_accept,
                save_dir, save_id):

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
        bic = np.log(n_subj * n_trials) * n_params + 2 * nll
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

    # Plot results
    if not run_on_cluster:
        print('Making and saving a plot!')
        if fit_mcmc:
            pm.traceplot(trace)

        if fit_map:
            n_subplots = len(map.keys())
            plt.figure(figsize=(5, 2 * n_subplots))
            for i, key in enumerate(map.keys()):
                plt.subplot(n_subplots, 1, i + 1)
                # plt.hist(map[key])
                # sns.kdeplot(map[key])
                sns.distplot(map[key])
                plt.title(key)
            plt.tight_layout()

        plt.savefig("{0}plot_{1}.png".format(save_dir, save_id))

    if fit_map:
        return nll, bic


# Determine which models to run
model_names = ['WSLS', 'strat']
for initialize in ['', 'i']:
    for states in ['', 'S', 'SS']:
        for model_name in ['ab', 'abc', 'abn', 'abcn', 'abcnnc']:
            model_names.append(model_name + states + initialize)

# Load data on which to run the model
n_subj, rewards, choices, group, n_groups, age_z = load_data(run_on_cluster)

# Run the model(s)
nll_bics = pd.DataFrame()
for model_name in model_names:

    # Create model
    model, n_params, n_trials, save_dir, save_id = create_model(
        choices=choices, rewards=rewards, group=group, model_name=model_name, n_subj=n_subj)

    # Fit parameters, calculate model fits, make plots, and save everything
    nll, bic = get_results(model, n_params, n_subj, n_trials,
                           fit_mcmc, fit_map,
                           n_samples, n_tune, n_chains, n_cores, target_accept,
                           save_dir, save_id)

    nll_bics = nll_bics.append([[model_name, n_params, nll, bic]])
    nll_bics.to_csv(save_dir + '/nll_bics_temp.csv', index=False)

nll_bics.columns = ['model_name', 'n_params', 'nll', 'bic']
nll_bics.to_csv(save_dir + '/nll_bics.csv', index=False)
