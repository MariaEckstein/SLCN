import glob
import datetime
import os

import numpy as np
import pandas as pd
import theano.tensor as T

import matplotlib.pyplot as plt
import seaborn as sns

from shared_modeling_simulation import get_paths
from shared_aliens import get_alien_paths


def load_aliens_data(run_on_cluster, fitted_data_name, param_names, file_name_suff, n_subj, n_trials, verbose):

    # Get data path and save path
    paths = get_alien_paths(run_on_cluster)
    if fitted_data_name == 'humans':
        data_dir = paths['human data']
        file_name_pattern = 'aliens*.csv'
    else:
        data_dir = paths['simulations']
        file_name_pattern = 'aliens*' + file_name_suff + '*.csv'

    # Prepare things for loading data
    filenames = glob.glob(data_dir + file_name_pattern)[:n_subj]
    assert len(filenames) > 0, "Error: There are no files with pattern {0} in {1}".format(file_name_pattern, data_dir)
    seasons = np.zeros((n_trials, len(filenames)))
    aliens = np.zeros(seasons.shape)
    actions = np.zeros(seasons.shape)
    rewards = np.zeros(seasons.shape)
    true_params = pd.DataFrame(np.ones((len(param_names), n_subj)),
                               index=['true_' + param_name for param_name in param_names])

    # Load data and bring in the right format
    for file_idx, filename in enumerate(filenames):
        agent_data = pd.read_csv(filename)
        if agent_data.shape[0] > n_trials:

            # Remove all rows that do not contain 1InitialLearning data (-> jsPysch format)
            agent_data = agent_data.rename(columns={'TS': 'context'})
            context_names = [str(TS) for TS in range(3)]
            item_names = range(3)
            agent_data = agent_data.loc[
                (agent_data['context'].isin(context_names)) &
                # TODO: remove following line; take care of missing data elegantly using masked numpy arrays
                (agent_data['item_chosen'].isin(item_names)) &
                (agent_data['phase'] == '1InitialLearning')]
            agent_data.index = range(agent_data.shape[0])

            seasons[:, file_idx] = np.array(agent_data['context'])[:n_trials]
            seasons = seasons.astype(int)
            aliens[:, file_idx] = np.array(agent_data['sad_alien'])[:n_trials]
            aliens = aliens.astype(int)
            actions[:, file_idx] = np.array(agent_data['item_chosen'])[:n_trials]
            # actions[np.isnan(actions)] = -999
            # actions_masked = np.ma.masked_values(actions, value=-999)
            actions = actions.astype(int)
            # actions_masked = actions_masked.astype(int)
            rewards[:, file_idx] = agent_data['reward'].tolist()[:n_trials]
            # rewards = (rewards - rewards.mean(axis=0, keepdims=True)) / rewards.std(axis=0, keepdims=True)  # TODO: subjwise z-scores
            # rewards[actions == -999] = -999
            # rewards_masked = np.ma.masked_values(rewards, value=999)
            # sID = filename[-7:-4]

            if fitted_data_name == 'simulations':
                true_params.ix[:, file_idx] = [agent_data[param_name][0] for param_name in param_names]

    # Remove excess columns (participants)
    seasons = np.delete(seasons, range(file_idx + 1, n_subj), 1)
    aliens = np.delete(aliens, range(file_idx + 1, n_subj), 1)
    rewards = np.delete(rewards, range(file_idx + 1, n_subj), 1)
    actions = np.delete(actions, range(file_idx + 1, n_subj), 1)

    n_subj = actions.shape[1]

    # Look at data
    print("Loaded {0} datasets with pattern {1} from {2}\n".format(n_subj, file_name_pattern, data_dir))
    if verbose:
        print("Choices - shape: {0}\n{1}\n".format(actions.shape, actions))
        print("Rewards - shape: {0}\n{1}\n".format(rewards.shape, rewards))

    return [n_subj, n_trials, seasons, aliens, actions, rewards, true_params]


def load_data(run_on_cluster, fitted_data_name='humans', n_groups='gender', kids_and_teens_only=False, adults_only=False, n_subj='all', n_trials=120, fit_slopes=False):

    # Get data path and save path
    paths = get_paths(run_on_cluster)
    if fitted_data_name == 'humans':
        data_dir = paths['human data']
        file_name_pattern = 'PS*.csv'
    else:
        learning_style = 'hierarchical'
        data_dir = paths['simulations']
        file_name_pattern = 'PS' + learning_style + '*.csv'

    # Prepare things for loading data
    print("Preparing to load {0} datasets with pattern {1} from {2}...\n".format(n_subj, file_name_pattern, data_dir))
    filenames = glob.glob(data_dir + file_name_pattern)
    if n_subj != 'all':
        filenames = filenames[:n_subj]
    else:
        n_subj = len(filenames)
    assert len(filenames) > 0, "Error: There are no files with pattern {0} in {1}".format(file_name_pattern, data_dir)
    choices = np.zeros((n_trials, len(filenames)))
    rewards = np.zeros(choices.shape)
    age = pd.DataFrame()

    # Load data and bring in the right format
    SLCNinfo = pd.read_csv(paths['SLCN info'])
    file_idx = 0
    for filename in filenames:
        agent_data = pd.read_csv(filename)
        if agent_data.shape[0] >= n_trials:
            choices[:, file_idx] = np.array(agent_data['selected_box'])[:n_trials]
            rewards[:, file_idx] = agent_data['reward'].tolist()[:n_trials]
            sID = agent_data['sID'][0]
            subj_age = SLCNinfo[SLCNinfo['ID'] == sID]['PreciseYrs'].values
            subj_gender = SLCNinfo[SLCNinfo['ID'] == sID]['Gender'].values
            subj_PDS = SLCNinfo[SLCNinfo['ID'] == sID]['PDS'].values
            subj_T1 = SLCNinfo[SLCNinfo['ID'] == sID]['T1'].values
            if not subj_age:
                subj_age = [np.nan]
            if not subj_gender:
                subj_gender = [np.nan]
            age = age.append([[sID, *subj_age, *subj_gender, *subj_PDS, *subj_T1]])
            file_idx += 1
        else:
            print("file {0} has only {2} rows (minimum is {1}) and will be excluded from analyses!".
                  format(filename, n_trials, agent_data.shape[0]))

    # Remove excess columns
    age.columns = ['sID', 'age', 'gender', 'PDS', 'T1']
    rewards = np.delete(rewards, range(file_idx, n_subj), 1)
    choices = np.delete(choices, range(file_idx, n_subj), 1)

    # Delete kid/teen or adult data sets
    if kids_and_teens_only:
        rewards = rewards[:, age['age'] <= 18]
        choices = choices[:, age['age'] <= 18]
        age = age[age['age'] <= 18]
    elif adults_only:
        rewards = rewards[:, age['age'] > 18]
        choices = choices[:, age['age'] > 18]
        age = age[age['age'] > 18]
    n_subj = choices.shape[1]

    # Get each participant's group assignment
    group = np.zeros(n_subj, dtype=int)
    if n_groups == 1:
        pass
    elif n_groups == 3:
        group[age['age'] > 12] = 1
        group[age['age'] > 17] = 2
    elif n_groups == 4:
        group[age['age'] > 11] = 1
        group[age['age'] > 15] = 2
        group[age['age'] > 20] = 3
    elif n_groups == 'gender':
        group = age['gender'] - 1  # gender is coded as 1, 2
        group[np.invert((group >= 0) * (group <= 1))] = np.nan  # exclude gender == 3 (non-gender-binary)
    else:
        raise(ValueError('n_groups can only be 1, 3, 4, or "gender"!'))

    # Find subjects that are missing data
    idxs_without_age = np.isnan(age['age'])
    idxs_without_gender = np.isnan(age['gender'])
    idxs_without_PDS = np.isnan(age['PDS'])
    idxs_without_T1 = np.isnan(age['T1'])

    # Find subjects with outlier performance (checked in R on 2019-06-11)
    # "Participants with n_switches < 5: 45, 102, 1004"
    # "Participants with mean_ACC < 0.58: 24, 45, 101, 102, 124"
    idxs_with_bad_perf = [id in [24, 45, 101, 102, 124, 1004] for id in age['sID']]
    print("Subjects {} are missing age and are removed from analyses!".format(
        age.loc[idxs_without_age, 'sID'].values))
    print("Subjects {} are missing gender and are removed from analyses!".format(
        age.loc[idxs_without_gender, 'sID'].values))
    print("Subjects {} have bad performance (n_switches < 5 | mean_ACC < 0.58) and are removed from analyses!".format(age.loc[idxs_with_bad_perf, 'sID'].values))

    # Remove marked subjects
    if fit_slopes and kids_and_teens_only:
        keep = np.invert(idxs_without_age | idxs_without_gender | idxs_with_bad_perf | idxs_without_T1 | idxs_without_PDS)
        print("Subjects {} are missing PDS and are removed from analyses!".format(
            age.loc[idxs_without_PDS, 'sID'].values))
        print(
            "Subjects {} are missing T1 and are removed from analyses!".format(age.loc[idxs_without_T1, 'sID'].values))
    elif fit_slopes and adults_only:
        keep = np.invert(idxs_without_age | idxs_without_gender | idxs_with_bad_perf | idxs_without_T1)
        print(
            "Subjects {} are missing T1 and are removed from analyses!".format(age.loc[idxs_without_T1, 'sID'].values))
    else:
        keep = np.invert(idxs_without_age | idxs_without_gender | idxs_with_bad_perf)
    n_subj = np.sum(keep)
    age = age[keep]
    group = group[list(keep)]
    n_groups = len(np.unique(group))
    rewards = rewards[:, keep]
    choices = choices[:, keep]
    print("Number of subjects after exluding: {0}; number of trials: {1}; (if hierarchical fitting, number of groups: {2})".format(n_subj, n_trials, n_groups))

    # z-score age
    age['age_z'] = (age['age'] - np.nanmean(age['age'])) / np.nanstd(age['age'])
    age['PDS_z'] = (age['PDS'] - np.nanmean(age['PDS'])) / np.nanstd(age['PDS'])
    age['T1_z'] = (age['T1'] - np.nanmean(age['T1'])) / np.nanstd(age['T1'])
    age['T1_log'] = np.log(age['T1'])
    age['T1_log_z'] = (age['T1_log'] - np.nanmean(age['T1_log'])) / np.nanstd(age['T1_log'])
    age['pymc3_idx'] = range(len(age))
    pd.DataFrame(age).to_csv(get_paths(run_on_cluster)['ages'], index=False)
    print("Saved ages.csv to {}".format(get_paths(run_on_cluster)['ages']))

    return [n_subj, rewards, choices, group, n_groups, age]


def get_save_dir_and_save_id(run_on_cluster, file_name_suff, fitted_data_name, n_samples):

    save_dir = get_paths(run_on_cluster)['fitting results']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    now = datetime.datetime.now()
    save_id = '_'.join([file_name_suff,
                        str(now.year), str(now.month), str(now.day), str(now.hour), str(now.minute),
                        fitted_data_name, 'n_samples' + str(n_samples)])

    return save_dir, save_id


def print_logp_info(model):

    print("Checking that none of the logp are -inf:")
    print("Test point: {0}".format(model.test_point))
    print("\tmodel.logp(model.test_point): {0}".format(model.logp(model.test_point)))

    for RV in model.basic_RVs:
        print("\tlogp of {0}: {1}".format(RV.name, RV.logp(model.test_point)))


def plot_gen_rec(param_names, gen_rec, save_name):

    plt.figure()
    for i, param_name in enumerate(param_names):

        # Plot fitted parameters against true parameters
        plt.subplot(3, 3, i + 1)
        sns.regplot(gen_rec.loc[param_name], gen_rec.loc['true_' + param_name], fit_reg=False)
        y_max = np.max(gen_rec.loc['true_' + param_name])
        plt.plot((0, y_max), (0, y_max))

    # Plot fitted alpha_high * beta_high against recovered alpha_high * beta_high
    plt.subplot(3, 3, 7)
    sns.regplot(gen_rec.loc['alpha_high'] * gen_rec.loc['beta_high'], gen_rec.loc['true_alpha_high'] * gen_rec.loc['true_beta_high'], fit_reg=False)
    y_max = np.max(gen_rec.loc['true_alpha_high'] * gen_rec.loc['true_beta_high'])
    plt.plot((0, y_max), (0, y_max))
    plt.xlabel('alpha_high * beta_high')
    plt.ylabel('true alpha_high * beta_high')

    # Plot fitted alpha * beta against recovered alpha * beta
    plt.subplot(3, 3, 8)
    sns.regplot(gen_rec.loc['alpha'] * gen_rec.loc['beta'], gen_rec.loc['true_alpha'] * gen_rec.loc['true_beta'], fit_reg=False)
    y_max = np.max(gen_rec.loc['true_alpha'] * gen_rec.loc['true_beta'])
    plt.plot((0, y_max), (0, y_max))
    plt.xlabel('alpha * beta')
    plt.ylabel('true alpha * beta')

    # Plot fitted alpha against fitted beta
    plt.subplot(3, 3, 9)
    sns.regplot(gen_rec.loc['alpha'], gen_rec.loc['beta'], fit_reg=False)
    if save_name:
        plt.savefig(save_name)
