import pickle
import os

import pandas as pd

from shared_modeling_simulation import get_paths
from shared_aliens import *
from AlienTask import Task

# Switches for this script
model_name = "flat_abf"
verbose = False
max_n_subj = 20  # must be > 1
# model_to_be_simulated = 'AliensFlat/flat_2018_8_8_17_36_humans_n_samples200aliens'
model_to_be_simulated = 'none'

# Get save path
save_dir = get_paths(False)['simulations']
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Type in some parameters
if model_to_be_simulated == 'none':
    n_subj = max_n_subj
    alpha = 0.12 * np.ones(n_subj)
    beta = 1 * np.ones(n_subj)
    forget = 0.003 * np.ones(n_subj)

# Load fitted parameters
else:
    parameter_dir = get_paths(run_on_cluster=False)['fitting results']
    print('Loading {0}{1}...\n'.format(parameter_dir, model_to_be_simulated))
    with open(parameter_dir + model_to_be_simulated + '.pickle', 'rb') as handle:
        data = pickle.load(handle)
        model_summary = data['summary']
        model = data['model']

    alpha_idx = [idx for idx in model_summary.index if 'alpha' in idx and '_mu' not in idx]
    alpha = model_summary.loc[alpha_idx[:max_n_subj], 'mean'].values

    beta_idx = [idx for idx in model_summary.index if 'beta' in idx and '_mu' not in idx]
    beta = model_summary.loc[beta_idx[:max_n_subj], 'mean'].values

    forget_idx = [idx for idx in model_summary.index if 'forget' in idx and '_mu' not in idx]
    forget = model_summary.loc[forget_idx[:max_n_subj], 'mean'].values

if verbose:
    print("Alphas: {0}".format(alpha.round(3)))
    print("Betas: {0}\n".format(beta.round(3)))
    print("Forgets: {0}".format(forget.round(3)))

# Get numbers of things
n_subj = len(alpha)
n_seasons, n_aliens, n_actions = 3, 4, 3
n_TS = n_seasons
n_alien_repetitions = np.array([13, 7, 7])  # InitialLearning, Refresher2, Refresher3
n_season_repetitions = np.array([3, 2, 2])  # InitialLearning, Refresher2, Refresher3

# Initialize task
task = Task(n_subj)
n_trials = task.get_trial_sequence("C:/Users/maria/MEGAsync/Berkeley/TaskSets/Data/version3.1/aliens139.csv")  # TODO: read in multiple files to have a different sequence for each simulated agent

# For saving data
seasons = np.zeros([n_trials, n_subj], dtype=int)
aliens = np.zeros([n_trials, n_subj], dtype=int)
actions = np.zeros([n_trials, n_subj], dtype=int)
rewards = np.zeros([n_trials, n_subj])
corrects = np.zeros([n_trials, n_subj])
p_norms = np.zeros([n_trials, n_subj, n_actions])
Q_lows = np.zeros([n_trials, n_subj, n_TS, n_aliens, n_actions])

# trials, subj = np.meshgrid(range(n_trials), range(n_subj))
# trials = trials.T
# subj = subj.T

# Adjust shapes for manipulations later-on
betas = np.repeat(beta, n_actions).reshape([n_subj, n_actions])  # for Q_sub
forgets = np.repeat(forget, n_TS * n_aliens * n_actions).reshape([n_subj, n_TS, n_aliens, n_actions])  # Q_low 1 trial

print('Simulating {0} agents on {1} trials.\n'.format(n_subj, n_trials))
action = np.full(n_subj, np.nan)
reward = np.full(n_subj, np.nan)

Q_low = alien_initial_Q * np.ones([n_subj, n_TS, n_aliens, n_actions])

for trial in range(np.sum(n_trials)):

    if verbose:
        print("\n\tTRIAL {0}".format(trial))

    # Observe stimuli
    season, alien = task.present_stimulus(trial)

    # Calculate action probabilities
    Q_sub = Q_low[np.arange(n_subj), season, alien]
    p_low = np.exp(betas * Q_sub)
    p_norm = np.array([p_low_subj / np.sum(p_low_subj) for p_low_subj in p_low])
    action = [np.random.choice(range(n_actions), p=p_low_subj) for p_low_subj in p_norm]
    reward, correct = task.produce_reward(action)

    # Update Q-values
    try:
        Q_low = update_Q_low_sim(season, alien, action, reward, Q_low, alpha, forgets, n_subj)
    except IndexError:
        print('Using alien_initial_Q!')

    # Print info
    if verbose:
        print("season:", season)
        print("alien:", alien)
        print("p_norm:", p_norm.round(3))
        print("action:", action)
        print("reward:", reward)
        print("Q_low:", Q_low.round(3))

    # Store trial data
    seasons[trial] = season
    aliens[trial] = alien
    actions[trial] = action
    rewards[trial] = reward
    corrects[trial] = correct
    # Q_lows[trial] = Q_low
    p_norms[trial] = p_norm

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
    file_name = save_dir + "aliens" + str(sID) + ".csv"
    print('Saving file {0}'.format(file_name))
    subj_data.to_csv(file_name)
