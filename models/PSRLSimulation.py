import pickle
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from shared_modeling_simulation import get_paths, update_Q, p_from_Q
from PStask import Task

# Switches for this script
verbose = False
make_plots = True
n_trials = 128  # humans: 128
param_names = np.array(['alpha', 'beta', 'nalpha', 'calpha', 'cnalpha', 'persev'])  #
n_subj = 2  # 233 as of 2018-10-03
n_sim_per_subj = 2
n_sim = n_sim_per_subj * n_subj
model_to_be_simulated = 'none'  # 'RL_3groups/abcnp_2018_11_15_11_21_humans_n_samples5000'
ages = pd.read_csv(get_paths(False)['ages'], index_col=0)

# Get save path
save_dir = get_paths(False)['simulations']
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

parameters = pd.DataFrame(columns=np.append(param_names, ['sID']))

# Type in some parameters
if model_to_be_simulated == 'none':
    parameters['alpha'] = 0.1 + 0.1 * np.random.rand(n_sim)
    parameters['eps'] = 0 * np.ones(n_sim)
    parameters['beta'] = 2 + 3 * np.random.rand(n_sim)
    # parameters['persev'] = 0.05 * np.random.rand(n_sim)
    # parameters['gamma'] = 0.8 + 0.2 * np.random.rand(n_sim)
    parameters['nalpha'] = 0.4 + 0.1 * np.random.rand(n_sim)
    parameters['calpha_sc'] = 0.9 * np.ones(n_sim)
    parameters['calpha'] = parameters['alpha'] * parameters['calpha_sc']
    parameters['cnalpha_sc'] = 0.9 * np.ones(n_sim)
    parameters['cnalpha'] = parameters['nalpha'] * parameters['cnalpha_sc']
    parameters['sID'] = range(n_sim)

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
        sample_params = pd.DataFrame(np.array([trace[param_name].squeeze()[rand_idx] for param_name in param_names]).T, columns=param_names)
        sample_params['sID'] = ages['sID']  # index of ages == PyMC's file order == samples_params' order
        parameters = pd.concat([parameters, sample_params], axis=0)

    if 'persev' not in parameters:
        parameters['persev'] = 0

    print('Model contains {0} participants, ages.csv {1}.'
          .format(trace[param_names[0]].shape[1], ages.shape[0]))

    # for param_name in param_names[param_names != 'eps']:
    #
    #     param_idx = [idx for idx in model_summary.index if param_name + '__' in idx]
    #     parameters[param_name] = model_summary.loc[param_idx[:n_subj], 'mean'].values
    # parameters = pd.DataFrame(np.tile(parameters, (n_sim_per_subj, 1)))
    # parameters = pd.DataFrame(np.array(parameters, dtype=float))
    # parameters.columns = param_names

print("Parameters: {0}".format(parameters.round(3)))

# Make sure that simulated agents get the same reward versions as humans
reward_versions = pd.read_csv(get_paths(run_on_cluster=False)['PS reward versions'], index_col=0)
assert np.all((reward_versions["sID"] % 4) == (reward_versions["rewardversion"]))

# Set up data frames
rewards = np.zeros((n_trials, n_sim_per_subj * n_subj), dtype=int)
choices = rewards.copy()  # human data: left choice==0; right choice==1 (verified in R script 18/11/17)
correct_boxes = rewards.copy()  # human data: left choice==0; right choice==1 (verified in R script 18/11/17)
ps_right = rewards.copy()
LLs = rewards.copy()
# Qs_trials = np.zeros((n_trials, 2, 2, 2, n_sim))
Qs_trials = np.zeros((n_trials, 2, n_sim))

# Initialize task
task_info_path = get_paths(run_on_cluster=False)['PS task info']
task = Task(task_info_path, n_sim_per_subj * n_subj, parameters['sID'])
LL = np.zeros(n_sim_per_subj * n_subj)

print('Simulating {0} agents ({2} simulations each) on {1} trials.\n'.format(n_subj, n_trials, n_sim_per_subj))
for trial in range(n_trials):

    if verbose:
        print("\tTRIAL {0}".format(trial))
    task.prepare_trial()

    # Update Q-values (starting on third trial)
    if trial <= 1:
        # Qs = 0.5 * np.ones((2, 2, 2, n_sim))  # shape: (n_prev_choice, n_prev_reward, n_choice, n_sim)
        Qs = 0.5 * np.ones((2, n_sim))  # shape: (n_choice, n_sim)
    else:
        # index = [choices[trial-2], rewards[trial-2], choices[trial-1], np.arange(n_sim)]
        # mirror_index = [1 - choices[trial-2], rewards[trial-2], 1 - choices[trial-1], np.arange(n_sim)]
        index = [choices[trial-1], np.arange(n_sim)]
        Qs = update_Q(
            choices[trial - 1], rewards[trial - 1],
            Qs,
            np.array(parameters['alpha']))  # , parameters['nalpha'], parameters['calpha'], parameters['cnalpha'])

    # Translate Q-values into action probabilities
    if verbose:
        print(np.round(Qs, 2))
    if trial == 0:
        p_right = 0.5 * np.ones(n_sim)
    else:
        # index_l = choices[trial-1], rewards[trial-1], 0, np.arange(n_sim)
        # index_r = choices[trial-1], rewards[trial-1], 1, np.arange(n_sim)
        index_l = 0, np.arange(n_sim)
        index_r = 1, np.arange(n_sim)
        p_right = p_from_Q(
            Qs,
            index_l, index_r,
            np.array(parameters['beta']), np.zeros(n_sim), verbose)

    # Select an action based on probabilities
    choice = np.random.binomial(n=1, p=p_right)  # produces "1" with p_right, and "0" with (1 - p_right)

    # Obtain reward
    reward = task.produce_reward(choice)

    # Update log-likelihood
    LL += np.log(p_right * choice + (1 - p_right) * (1 - choice))

    if verbose:
        print("Choice:", choice)
        print("Reward:", reward)
        print("LL:", LL)

    # Store trial data
    ps_right[trial] = p_right
    choices[trial] = choice
    rewards[trial] = reward
    correct_boxes[trial] = task.correct_box
    LLs[trial] = LL
    Qs_trials[trial] = Qs
    # Qs_left[trial] = Q_left
    # Qs_right[trial] = Q_right
    # Qs_L1L_R1R[trial] = Q_L1L_R1R
    # Qs_L0L_R0R[trial] = Q_L0L_R0R
    # Qs_L1R_R1L[trial] = Q_L1R_R1L
    # Qs_L0R_R0L[trial] = Q_L0R_R0L

# Plot development of Q-values
if make_plots:
    for subj in range(min(n_sim, 5)):
        plt.figure()
        # plt.plot(Qs_trials[:, 0, 0, 0, subj], label='L0L')
        # plt.plot(Qs_trials[:, 1, 0, 1, subj], label='R0R')
        # plt.plot(Qs_trials[:, 0, 1, 0, subj], label='L1L')
        # plt.plot(Qs_trials[:, 1, 1, 1, subj], label='R1R')
        # plt.plot(Qs_trials[:, 0, 1, 1, subj], label='L1R')
        # plt.plot(Qs_trials[:, 1, 1, 0, subj], label='R1L')
        # plt.plot(Qs_trials[:, 0, 0, 1, subj], label='L0R')
        # plt.plot(Qs_trials[:, 1, 0, 0, subj], label='R0L')
        plt.plot(Qs_trials[:, 0, subj], label='L')
        plt.plot(Qs_trials[:, 1, subj], label='R')
        plt.ylim((0, 1))
        plt.legend()
    plt.show()

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
    # subj_data["Q_left"] = Qs_left[:, i]
    # subj_data["Q_right"] = Qs_right[:, i]
    # for param_name in param_names:
    #     subj_data[param_name] = np.array(parameters.loc[i, param_name])[version]

    # Save to disc
    file_name = save_dir + "PSRL_{0}_{1}.csv".format(int(sID), version)
    subj_data.to_csv(file_name)
    # print('Saving file {0}'.format(file_name))

print("Ran and saved {0} simulations ({1} * {2}) to {3}!".
      format(len(parameters['sID']), n_subj, n_sim_per_subj, file_name))
