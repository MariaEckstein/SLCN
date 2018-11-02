import pickle
import os

import pandas as pd
import numpy as np

from shared_modeling_simulation import get_alien_paths
from shared_aliens import alien_initial_Q, update_Qs_sim
from AlienTask import Task


# Switches for this script
model_name = "soft"
verbose = False
n_subj = 31
n_sim_per_subj = 1
start_id = 0
param_names = np.array(['alpha', 'beta', 'forget', 'alpha_high', 'beta_high', 'forget_high'])
fake_data = False
model_to_be_simulated = "specify"  # "MSE"  # "MCMC" "specify"
# model_name = "/AliensMSEFitting/18-10-14/f_['alpha' 'beta' 'forget']_[[ 1 10  1]]_2018_10_14_9_47"  # 'Aliens/max_abf_2018_10_10_18_7_humans_n_samples10'  #
# Get save path
save_dir = get_alien_paths(False)['simulations']
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Get parameters
parameters = pd.DataFrame(columns=np.append(param_names, ['sID']))
parameter_dir = get_alien_paths(run_on_cluster=False)['fitting results']

if model_to_be_simulated == 'specify':

    parameters['alpha'] = 0.01 + 0.05 * np.random.rand(n_subj)  # 0 < alpha < 0.2
    parameters['beta'] = 6.5 + np.random.rand(n_subj)  # 1 < beta < 2
    parameters['forget'] = 0.005 + 0.01 * np.random.rand(n_subj)  # 0 < forget < 0.1

    parameters['alpha_high'] = 0.3 + 0.1 * np.random.rand(n_subj)  # 0 < alpha_high < 0.2
    parameters['beta_high'] = 4.5 + np.random.rand(n_subj)  # 1 < beta < 2
    parameters['forget_high'] = 0.2 + 0.1 * np.random.rand(n_subj)

    parameters['sID'] = range(n_subj)

# Load fitted parameters
elif model_to_be_simulated == 'MSE':

    parameters = pd.read_csv(parameter_dir + '/ten_best_{}.csv'.format(model_name), index_col=0)
    n_subj = min(n_subj, parameters.shape[0])
    parameters = parameters[:n_subj]
    parameters['sID'] = range(n_subj)

    if 'alpha_high' not in parameters:
        parameters['alpha_high'] = parameters['alpha'].copy()
    if 'beta_high' not in parameters:
        parameters['beta_high'] = parameters['beta'].copy()
    if 'forget_high' not in parameters:
        parameters['forget_high'] = parameters['forget'].copy()

elif model_to_be_simulated == 'MCMC':

    print('Loading {0}{1}.\n'.format(parameter_dir, model_name))
    with open(parameter_dir + model_name + '.pickle', 'rb') as handle:
        data = pickle.load(handle)
        model_summary = data['summary']
        model = data['model']

    model_summary = pd.read_csv(parameter_dir + model_name + '_summary.csv', index_col=0)
    for param_name in param_names:
        param_idx = [idx for idx in model_summary.index if param_name + '__' in idx]
        parameters[param_name] = model_summary.loc[param_idx[:n_subj], 'mean'].values
    parameters['sID'] = range(n_subj)
    parameters = pd.DataFrame(np.tile(parameters, (n_sim_per_subj, 1)))
    parameters = pd.DataFrame(np.array(parameters, dtype=float))
    parameters.columns = np.append(param_names, ['sID'])

if verbose:
    print("Parameters: {}".format(parameters.round(3)))

# Parameter shapes
n_sim = n_subj * n_sim_per_subj
beta_shape = (n_sim, 1)  # Q_low_sub.shape -> [n_subj, n_actions]
beta_high_shape = (n_sim, 1)  # Q_high_sub.shape -> [n_subj, n_TS]
forget_high_shape = (n_sim, 1, 1)  # -> [n_subj, n_seasons, n_TS]
forget_shape = (n_sim, 1, 1, 1)  # Q_low[0].shape -> [n_subj, n_TS, n_aliens, n_actions]

# Get numbers of things
n_seasons, n_aliens, n_actions = 3, 4, 3
n_TS = n_seasons
n_alien_repetitions = np.array([13, 7, 7])  # InitialLearning, Refresher2, Refresher3
n_season_repetitions = np.array([3, 2, 2])  # InitialLearning, Refresher2, Refresher3

# Initialize task
task = Task(n_subj)
n_trials, _, _ = task.get_trial_sequence("C:/Users/maria/MEGAsync/Berkeley/TaskSets/Data/version3.1/",
                                         n_subj, n_sim_per_subj, range(n_subj), fake_data)
print("n_trials", n_trials)

# For saving data
seasons = np.zeros([n_trials, n_sim], dtype=int)
TSs = np.zeros([n_trials, n_sim], dtype=int)
aliens = np.zeros([n_trials, n_sim], dtype=int)
actions = np.zeros([n_trials, n_sim], dtype=int)
rewards = np.zeros([n_trials, n_sim])
corrects = np.zeros([n_trials, n_sim])
p_lows = np.zeros([n_trials, n_sim, n_actions])
# Q_lows = np.zeros([n_trials, n_sim, n_TS, n_aliens, n_actions])
Q_highs = np.zeros([n_trials, n_sim])

print('Simulating {0} {2} agents on {1} trials.\n'.format(n_sim, n_trials, model_name))

Q_low = alien_initial_Q * np.ones([n_sim, n_TS, n_aliens, n_actions])
Q_high = alien_initial_Q * np.ones([n_sim, n_seasons, n_TS])

# Bring parameters into the right shape
alpha = parameters['alpha'].values
beta = parameters['beta'].values.reshape(beta_shape)
forget = parameters['forget'].values.reshape(forget_shape)
alpha_high = parameters['alpha_high'].values
beta_high = parameters['beta_high'].values.reshape(beta_high_shape)
forget_high = parameters['forget_high'].values.reshape(forget_high_shape)

for trial in range(np.sum(n_trials)):

    # Observe stimuli
    season, alien = task.present_stimulus(trial)

    # Print info
    if verbose:
        print("\n\tTRIAL {0}".format(trial))
        print("season:", season)
        print("alien:", alien)

    # Update Q-values
    [Q_low, Q_high, TS, action, correct, reward, p_low] =\
        update_Qs_sim(season, alien,
                           Q_low, Q_high,
                           beta, beta_high, alpha, alpha_high, forget, forget_high,
                           n_sim, n_actions, n_TS, task, verbose=verbose)

    # Store trial data
    seasons[trial] = season
    TSs[trial] = TS
    aliens[trial] = alien
    actions[trial] = action
    rewards[trial] = reward
    corrects[trial] = correct
    # Q_lows[trial] = Q_low
    Q_highs[trial] = Q_high[np.arange(n_sim), season, TS]
    p_lows[trial] = p_low

# Save data
for sID in range(n_sim):

    agent_ID = sID + start_id
    # Create pandas DataFrame
    subj_data = pd.DataFrame()
    subj_data["context"] = seasons[:, sID]
    subj_data["TS_chosen"] = TSs[:, sID]
    subj_data["sad_alien"] = aliens[:, sID]
    subj_data["item_chosen"] = actions[:, sID]
    subj_data["reward"] = rewards[:, sID]
    subj_data["correct"] = corrects[:, sID]
    subj_data["trial_type"] = "feed-aliens"
    subj_data["phase"] = np.array(task.phase).astype(str)
    subj_data["trial_index"] = np.arange(n_trials)
    # subj_data["p_low"] = p_norms[:, sID]
    subj_data["sID"] = agent_ID
    subj_data["block.type"] = "normal"
    subj_data["model_name"] = model_name
    for param_name in param_names:
        subj_data[param_name] = np.array(parameters.loc[sID, param_name])
    # subj_data["alpha"], subj_data["beta"], subj_data["forget"] = alpha[sID], beta.flatten()[sID], forget.flatten()[sID]
    # subj_data["alpha_high"], subj_data["beta_high"], subj_data["forget_high"] = alpha_high[sID], beta_high.flatten()[sID], forget_high.flatten()[sID]
    # subj_data["Q_low"] = Q_lows[:, sID]
    subj_data["Q_TS"] = Q_highs[:, sID]

    # Save to disc
    file_name = save_dir + "aliens_" + model_name + '_' + str(agent_ID) + ".csv"
    print('Saving file {0}'.format(file_name))
    subj_data.to_csv(file_name)
