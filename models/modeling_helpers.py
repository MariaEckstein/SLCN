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


def get_population_level_priors():
    # Get population-level priors (as un-informative as possible)

    return {

        # RL and Bayes
        'eps_mu': pm.Uniform('eps_mu', lower=0, upper=1),
        'eps_sd': 0.1,
        'beta_mu': pm.Lognormal('beta_mu', mu=0, sd=1),
        'beta_sd': 3,  # pm.HalfNormal('beta_sd', sd=0.1)

        # Bayes only
        'p_switch_mu': pm.Uniform('p_switch_mu', lower=0, upper=1),
        'p_switch_sd': 0.1,  # pm.HalfNormal('p_switch_sd', sd=0.1)
        'p_reward_mu': pm.Uniform('p_reward_mu', lower=0, upper=1),
        'p_reward_sd': 0.1,  # pm.HalfNormal('p_reward_sd', sd=0.1)
        'p_noisy_mu': pm.Uniform('p_noisy_mu', lower=0, upper=1),
        'p_noisy_sd': 0.1,  # pm.HalfNormal('p_noisy_sd', sd=0.1)

        # RL only
        'alpha_mu': pm.Uniform('alpha_mu', lower=0, upper=1),
        'alpha_sd': 0.1,  # pm.HalfNormal('alpha_sd', sd=0.1)
        'calpha_sc_mu': pm.Uniform('calpha_sc_mu', lower=0, upper=1),
        'calpha_sc_sd': 0.1  # pm.HalfNormal('calpha_sc_sd', sd=0.1)
    }


def get_slopes(fit_slopes):
    if fit_slopes:
        return {
            'eps_sl': pm.Uniform('eps_sl', lower=-1, upper=1),
            'beta_sl': pm.Uniform('beta_sl', lower=-1, upper=1),
            'p_switch_sl': pm.Uniform('p_switch_sl', lower=-1, upper=1),
            'p_reward_sl': pm.Uniform('p_reward_sl', lower=-1, upper=1),
            'alpha_sl': pm.Uniform('alpha_sl', lower=0, upper=1),
            'calpha_sl': pm.Uniform('calpha_sl', lower=0, upper=1)
        }

    else:
        return {
            'eps_sl': T.as_tensor_variable(0.),
            'beta_sl': T.as_tensor_variable(0.),
            'p_switch_sl': T.as_tensor_variable(0.),
            'p_reward_sl': T.as_tensor_variable(0.),
            'alpha_sl': T.as_tensor_variable(0.),
            'calpha_sl': T.as_tensor_variable(0.)
        }
