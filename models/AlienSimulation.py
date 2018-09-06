import pickle
import os

import pandas as pd

from shared_modeling_simulation import get_paths
from shared_aliens import *
from AlienTask import Task

# Switches for this script
model_name = "f"
verbose = False
max_n_subj = 20  # must be > 1
# model_to_be_simulated = 'AliensFlat/flat_tilde_ta80_abf_2018_8_10_13_52_humans_n_samples300aliens'
model_to_be_simulated = 'none'

# Get save path
save_dir = get_paths(False)['simulations']
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Type in some parameters
if model_to_be_simulated == 'none':
    n_subj = max_n_subj
    alpha = 0.2 * np.ones(n_subj)
    alpha_high = 0.1 * np.ones(n_subj)
    beta = 1.5 * np.ones(n_subj)
    beta_high = beta.copy()
    forget = 0.001 * np.ones(n_subj)  # np.zeros(n_subj)  #
    forget_high = 0.005 * np.ones(n_subj)  # np.zeros(n_subj)  #

# Load fitted parameters
else:
    parameter_dir = get_paths(run_on_cluster=False)['fitting results']
    print('Loading {0}{1}...\n'.format(parameter_dir, model_to_be_simulated))
    with open(parameter_dir + model_to_be_simulated + '.pickle', 'rb') as handle:
        data = pickle.load(handle)
        model_summary = data['summary']
        model = data['model']

    alpha_idx = [idx for idx in model_summary.index if 'alpha' in idx
                 and '_mu' not in idx and 'sd' not in idx and 'high' not in idx and 'matt' not in idx]
    alpha = model_summary.loc[alpha_idx[:max_n_subj], 'mean'].values
    alpha_high_idx = [idx for idx in model_summary.index if 'alpha_high' in idx
                      and '_mu' not in idx and 'sd' not in idx and 'matt' not in idx]
    alpha_high = model_summary.loc[alpha_high_idx[:max_n_subj], 'mean'].values

    beta_idx = [idx for idx in model_summary.index if 'beta' in idx
                and '_mu' not in idx and 'sd' not in idx and 'high' not in idx and 'matt' not in idx]
    beta = model_summary.loc[beta_idx[:max_n_subj], 'mean'].values
    beta_high_idx = [idx for idx in model_summary.index if 'beta_high' in idx
                     and '_mu' not in idx and 'sd' not in idx and 'matt' not in idx]
    beta_high = model_summary.loc[beta_high_idx[:max_n_subj], 'mean'].values

    forget_idx = [idx for idx in model_summary.index if 'forget' in idx
                  and '_mu' not in idx and 'sd' not in idx and 'high' not in idx and 'matt' not in idx]
    forget = model_summary.loc[forget_idx[:max_n_subj], 'mean'].values
    forget_high_idx = [idx for idx in model_summary.index if 'forget_high' in idx
                       and '_mu' not in idx and '_sd' not in idx and 'matt' not in idx]
    forget_high = model_summary.loc[forget_high_idx[:max_n_subj], 'mean'].values
    if not forget_high:
        forget_high = forget.copy()

if verbose:
    print("Alpha: {0}".format(alpha.round(3)))
    print("Alpha_high: {0}".format(alpha_high.round(3)))
    print("Beta: {0}".format(beta.round(3)))
    print("Beta_high: {0}".format(beta_high.round(3)))
    print("Forget: {0}".format(forget.round(3)))
    print("Forget_high: {0}".format(forget_high.round(3)))

# Get numbers of things
n_subj = len(alpha)
n_seasons, n_aliens, n_actions = 3, 4, 3
n_TS = n_seasons
n_alien_repetitions = np.array([13, 7, 7])  # InitialLearning, Refresher2, Refresher3
n_season_repetitions = np.array([3, 2, 2])  # InitialLearning, Refresher2, Refresher3

# Initialize task
task = Task(n_subj)
n_trials = task.get_trial_sequence("C:/Users/maria/MEGAsync/Berkeley/TaskSets/Data/version3.1/", n_subj)

# For saving data
seasons = np.zeros([n_trials, n_subj], dtype=int)
aliens = np.zeros([n_trials, n_subj], dtype=int)
actions = np.zeros([n_trials, n_subj], dtype=int)
rewards = np.zeros([n_trials, n_subj])
corrects = np.zeros([n_trials, n_subj])
p_lows = np.zeros([n_trials, n_subj, n_actions])
Q_lows = np.zeros([n_trials, n_subj, n_TS, n_aliens, n_actions])

# Adjust shapes for manipulations later-on
betas = np.repeat(beta, n_actions).reshape([n_subj, n_actions])  # for Q_sub
beta_highs = np.repeat(beta_high, n_TS).reshape([n_subj, n_TS])  # Q_high_sub.shape
forgets = np.repeat(forget, n_TS * n_aliens * n_actions).reshape([n_subj, n_TS, n_aliens, n_actions])  # Q_low 1 trial
forget_highs = np.repeat(forget_high, n_seasons * n_TS).reshape([n_subj, n_seasons, n_TS])  # Q_high for 1 trial

print('Simulating {0} agents on {1} trials.\n'.format(n_subj, n_trials))

Q_low = alien_initial_Q * np.ones([n_subj, n_TS, n_aliens, n_actions])
Q_high = alien_initial_Q * np.ones([n_subj, n_seasons, n_TS])

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
                      betas, beta_highs, alpha, alpha_high, forgets, forget_highs,
                      n_subj, n_actions, task,
                      verbose=verbose)

    # Store trial data
    seasons[trial] = season
    aliens[trial] = alien
    actions[trial] = action
    rewards[trial] = reward
    corrects[trial] = correct
    # Q_lows[trial] = Q_low
    p_lows[trial] = p_low

# Save data
for sID in range(n_subj):

    # Create pandas DataFrame
    subj_data = pd.DataFrame()
    subj_data["context"] = seasons[:, sID]
    subj_data["sad_alien"] = aliens[:, sID]
    subj_data["item_chosen"] = actions[:, sID]
    subj_data["reward"] = rewards[:, sID]
    subj_data["correct"] = corrects[:, sID]
    subj_data["trial_type"] = "feed-aliens"
    subj_data["phase"] = np.array(task.phase).astype(str)
    subj_data["trial_index"] = np.arange(n_trials)
    # subj_data["p_low"] = p_norms[:, sID]
    subj_data["sID"] = sID
    subj_data["block.type"] = "normal"
    subj_data["model_name"] = model_name
    subj_data["alpha"], subj_data["beta"], subj_data["forget"] = alpha[sID], beta[sID], forget[sID]
    # subj_data["Q_low"] = Q_lows[:, sID]

    # Save to disc
    file_name = save_dir + "aliens_" + model_name + str(sID) + ".csv"
    print('Saving file {0}'.format(file_name))
    subj_data.to_csv(file_name)
