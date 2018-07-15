import glob

import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as T

from shared_modeling_simulation import get_paths


def load_data(run_on_cluster, fitted_data_name, kids_and_teens_only, adults_only, verbose):
    # Get data path and save path
    paths = get_paths(run_on_cluster)
    if fitted_data_name == 'humans':
        data_dir = paths['human data']
        file_name_pattern = 'PS*.csv'
        n_trials = 128
        n_subj = 500
    else:
        learning_style = 'hierarchical'
        data_dir = paths['simulations']
        file_name_pattern = 'PS' + learning_style + '*.csv'
        n_trials = 200
        n_subj = 50

    # Prepare things for loading data
    filenames = glob.glob(data_dir + file_name_pattern)[:n_subj]
    assert len(filenames) > 0, "Error: There are no files with pattern {0} in {1}".format(file_name_pattern, data_dir)
    choices = np.zeros((n_trials, len(filenames)))
    rewards = np.zeros(choices.shape)
    ages = np.zeros(n_subj)

    # Load data and bring in the right format
    SLCNinfo = pd.read_csv(paths['ages file name'])
    for file_idx, filename in enumerate(filenames):
        agent_data = pd.read_csv(filename)
        if agent_data.shape[0] > n_trials:
            choices[:, file_idx] = np.array(agent_data['selected_box'])[:n_trials]
            rewards[:, file_idx] = agent_data['reward'].tolist()[:n_trials]
            sID = agent_data['sID'][0]
            ages[file_idx] = SLCNinfo[SLCNinfo['ID'] == sID]['PreciseYrs'].values

    # Remove excess columns
    rewards = np.delete(rewards, range(file_idx + 1, n_subj), 1)
    choices = np.delete(choices, range(file_idx + 1, n_subj), 1)
    ages = ages[:file_idx + 1]

    # Delete kid/teen or adult data sets
    if kids_and_teens_only:
        rewards = rewards[:, ages <= 18]
        choices = choices[:, ages <= 18]
        ages = ages[ages <= 18]
    elif adults_only:
        rewards = rewards[:, ages > 18]
        choices = choices[:, ages > 18]
        ages = ages[ages > 18]

    n_subj = choices.shape[1]

    # Look at data
    print("Loaded {0} datasets with pattern {1} from {2}...\n".format(n_subj, file_name_pattern, data_dir))
    if verbose:
        print("Choices - shape: {0}\n{1}\n".format(choices.shape, choices))
        print("Rewards - shape: {0}\n{1}\n".format(rewards.shape, rewards))

    return n_subj, rewards, choices, ages


def get_population_level_priors(model_name):

    # Get population-level priors (as un-informative as possible)
    prior_dict = dict()
    # prior_dict = {'eps_mu': pm.Uniform('eps_mu', lower=0, upper=1),
    #               'beta_mu': pm.Lognormal('beta_mu', mu=0, sd=3),
    #               'eps_sd': pm.HalfNormal('eps_sd', sd=0.2),
    #               'beta_sd': T.as_tensor_variable(2.)}  # pm.HalfNormal('beta_sd', sd=3)}

    if model_name == 'RL':
        for par_name in ['alpha_mu', 'calpha_sc_mu']:
            prior_dict.update({par_name: pm.Uniform(par_name, lower=0, upper=1)})
        for par_name in ['alpha_sd', 'calpha_sc_sd']:
            prior_dict.update({par_name: T.as_tensor_variable(0.5)})  # pm.HalfNormal(par_name, sd=0.3)})

    elif model_name == 'Bayes':
        for par_name in ['p_switch_mu', 'p_reward_mu']:
            prior_dict.update({par_name: pm.Uniform(par_name, lower=0, upper=1)})
        for par_name in ['p_switch_sd', 'p_reward_sd']:
            prior_dict.update({par_name: T.as_tensor_variable(0.5)})  # pm.HalfNormal(par_name, sd=0.3)})

    return prior_dict


def get_slopes(fit_slopes, model_name):

    slopes_dict = dict()
    if fit_slopes:

        # Slopes can vary between -1 and 1
        for par_name in ['eps_sl', 'beta_sl']:
            slopes_dict.update({par_name: pm.Uniform(par_name, lower=-1, upper=1)})

        if model_name == 'RL':
            for par_name in ['alpha_sl', 'calpha_sc_sl']:
                slopes_dict.update({par_name: pm.Uniform(par_name, lower=-1, upper=1)})

        elif model_name == 'Bayes':
            for par_name in ['p_switch_sl', 'p_reward_sl']:
                slopes_dict.update({par_name: pm.Uniform(par_name, lower=-1, upper=1)})

    else:

        # All slopes are 0
        for par_name in ['eps_sl', 'beta_sl']:
            slopes_dict.update({par_name: T.as_tensor_variable(0.)})

        if model_name == 'RL':
            for par_name in ['alpha_sl', 'calpha_sc_sl']:
                slopes_dict.update({par_name: T.as_tensor_variable(0.)})

        elif model_name == 'Bayes':
            for par_name in ['p_switch_sl', 'p_reward_sl']:
                slopes_dict.update({par_name: T.as_tensor_variable(0.)})

    return slopes_dict


def get_tilde_pars(model_name, n_subj):

    tilde_dict = dict()
    for par_name in ['eps_tilde', 'beta_tilde']:
        tilde_dict.update({par_name: pm.Normal(par_name, mu=0, sd=0.2, shape=n_subj)})

    if model_name == 'RL':
        for par_name in ['alpha_tilde', 'calpha_sc_tilde']:
            tilde_dict.update({par_name: pm.Normal(par_name, mu=0, sd=0.2, shape=n_subj)})

    elif model_name == 'Bayes':
        for par_name in ['p_switch_tilde', 'p_reward_tilde']:
            tilde_dict.update({par_name: pm.Normal(par_name, mu=0, sd=0.2, shape=n_subj)})

    return tilde_dict

# def bound_variables(model_name):
#
#     bounded_dict = dict()
#     for par_name in ['eps_min', 'beta_min']:
#         bounded_dict.update({par_name: pm.Potential(par_name, T.switch())})
#     if model_name == 'RL':

