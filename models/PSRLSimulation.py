import pickle
import os

import pandas as pd
import numpy as np

from shared_modeling_simulation import get_paths, update_Q, p_from_Q
from PStask import Task

# Switches for this script
verbose = False
n_trials = 128  # humans: 128
n_sim_per_subj = 10
param_names = np.array(['alpha', 'beta', 'nalpha', 'calpha', 'cnalpha'])
n_subj = 233  # 233 as of 2018-10-03
max_n_subj = n_sim_per_subj * n_subj
model_to_be_simulated = 'RL_3groups/abcn_2018_10_3_19_35_humans_n_samples5000'  # 'none'  #
ages = pd.read_csv(get_paths(False)['ages'], index_col=0)

# Get save path
save_dir = get_paths(False)['simulations']
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

parameters = pd.DataFrame(columns=np.append(param_names, ['sID']))

# Type in some parameters
if model_to_be_simulated == 'none':
    parameters['alpha'] = np.random.rand(n_subj)
    parameters['eps'] = 0 * np.ones(n_subj)
    parameters['beta'] = 1 + 3 * np.random.rand(n_subj)
    parameters['nalpha'] = np.random.rand(n_subj)
    parameters['calpha_sc'] = 0 * np.ones(n_subj)
    parameters['calpha'] = parameters['alpha'] * parameters['calpha_sc']
    parameters['cnalpha_sc'] = 0 * np.ones(n_subj)
    parameters['cnalpha'] = parameters['nalpha'] * parameters['cnalpha_sc']
    parameters['sID'] = range(n_subj)

# Load fitted parameters
else:
    parameter_dir = get_paths(run_on_cluster=False)['fitting results']
    print('Loading {0}{1}.\n'.format(parameter_dir, model_to_be_simulated))
    # model_summary = pd.read_csv(parameter_dir + model_to_be_simulated + '_summary.csv', index_col=0)

    with open(parameter_dir + model_to_be_simulated + '.pickle', 'rb') as handle:
        data = pickle.load(handle)
        trace = data['trace']
        model = data['model']
        model_summary = data['summary']

    for sim_id in range(n_sim_per_subj):
        rand_idx = np.random.randint(0, trace[param_names[0]].shape[0])
        sample_params = pd.DataFrame(np.array([trace[param_name][rand_idx] for param_name in param_names]).T, columns=param_names)
        sample_params['sID'] = ages['sID']  # index of ages == PyMC's file order == samples_params' order
        parameters = pd.concat([parameters, sample_params], axis=0)

    print('Model contains {0} participants, ages.csv {1}.'
          .format(trace[param_names[0]].shape[1], ages.shape[0]))

    # for param_name in param_names[param_names != 'eps']:
    #
    #     param_idx = [idx for idx in model_summary.index if param_name + '__' in idx]
    #     parameters[param_name] = model_summary.loc[param_idx[:n_subj], 'mean'].values
    # parameters = pd.DataFrame(np.tile(parameters, (n_sim_per_subj, 1)))
    # parameters = pd.DataFrame(np.array(parameters, dtype=float))
    # parameters.columns = param_names

if verbose:
    print("Parameters: {0}".format(parameters.round(3)))

# Make sure that simulated agents get the same reward versions as humans
reward_versions = pd.read_csv(get_paths(run_on_cluster=False)['PS reward versions'], index_col=0)
assert np.all((reward_versions["sID"] % 4) == (reward_versions["rewardversion"]))

# Set up data frames
rewards = np.zeros((n_trials, n_sim_per_subj * n_subj))
choices = np.zeros(rewards.shape)
correct_boxes = np.zeros(rewards.shape)
ps_right = np.zeros(rewards.shape)
LLs = np.zeros(rewards.shape)

Qs_left = np.zeros(rewards.shape)
Qs_right = np.zeros(rewards.shape)

# Initialize task
task_info_path = get_paths(run_on_cluster=False)['PS task info']
task = Task(task_info_path, n_sim_per_subj * n_subj, parameters['sID'])
LL = np.zeros(n_sim_per_subj * n_subj)

print('Simulating {0} agents ({2} simulations each) on {1} trials.\n'.format(n_subj, n_trials, n_sim_per_subj))
for trial in range(n_trials):

    if verbose:
        print("\tTRIAL {0}".format(trial))
    task.prepare_trial()

    # Translate Q-values into action probabilities, make a choice, obtain reward, update Q-values
    try:
        Q_left, Q_right = update_Q(reward, choice, Q_left, Q_right, parameters['alpha'], parameters['nalpha'], parameters['calpha'], parameters['cnalpha'])
    except NameError:
        print('Initializing Q to 0.5!')
        Q_left, Q_right = 0.5 * np.ones(n_sim_per_subj * n_subj), 0.5 * np.ones(n_sim_per_subj * n_subj)

    p_right = p_from_Q(Q_left, Q_right, parameters['beta'], np.zeros(max_n_subj))
    choice = np.random.binomial(n=1, p=p_right)
    reward = task.produce_reward(choice)
    LL += np.log(p_right * choice + (1 - p_right) * (1 - choice))

    if verbose:
        print("p_right:", p_right.round(3))
        print("Choice:", choice)
        print("Reward:", reward)
        print("Q_left:", Q_left.round(3))
        print("Q_right:", Q_right.round(3))
        print("LL:", LL)

    # Store trial data
    ps_right[trial] = p_right
    choices[trial] = choice
    rewards[trial] = reward
    correct_boxes[trial] = task.correct_box
    LLs[trial] = LL
    Qs_left[trial] = Q_left
    Qs_right[trial] = Q_right

# Save data
for i, sID in enumerate(parameters['sID']):

    version = int(np.floor(i / n_subj))

    # Create pandas DataFrame
    subj_data = pd.DataFrame()
    subj_data["selected_box"] = choices[:, i]
    subj_data["reward"] = rewards[:, i]
    subj_data["correct_box"] = correct_boxes[:, i]
    subj_data["p_right"] = ps_right[:, i]
    subj_data["sID"] = sID
    subj_data["version"] = version
    subj_data["LL"] = LLs[:, i]
    subj_data["Q_left"] = Qs_left[:, i]
    subj_data["Q_right"] = Qs_right[:, i]
    # for param_name in param_names:
    #     subj_data[param_name] = np.array(parameters.loc[i, param_name])[version]

    # Save to disc
    file_name = save_dir + "PSRL_{0}_{1}.csv".format(int(sID), version)
    # print('Saving file {0}'.format(file_name))
    subj_data.to_csv(file_name)

print("Ran and saved {0} simulations ({1} * {2}) to {3}!".
      format(len(parameters['sID']), n_subj, n_sim_per_subj, file_name))
