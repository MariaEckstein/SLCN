import pickle
import os

import pandas as pd
import numpy as np

from shared_modeling_simulation import get_paths, get_likelihoods, post_from_lik
from PStask import Task

# Switches for this script
verbose = False
n_trials = 128  # humans: 128
param_names = np.array(['beta', 'p_switch', 'p_reward', 'persev'])  #
n_subj = 233  # 233 as of 2018-10-03
n_sim_per_subj = 2
n_sim = n_sim_per_subj * n_subj
# TODO: comment out `p_right = 1 / (1 + np.exp(-beta * (p_right - (1 - p_right))))`
# TODO: in shared_mod_sim when running swirew model (no beta)!
model_to_be_simulated = 'none'  # 'Bayes_3groups/betperswirew_2018_11_13_16_26_humans_n_samples5000'  # 'Bayes_add_persev/betperswirew_2018_11_13_16_26_humans_n_samples5000'  # 'Bayes_3groups/swirew_2018_10_10_17_29_humans_n_samples5000'  # 'none'  #
ages = pd.read_csv(get_paths(False)['ages'], index_col=0)

# Get save path
save_dir = get_paths(False)['simulations']
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

parameters = pd.DataFrame(columns=np.append(param_names, ['sID']))

# Type in some parameters
if model_to_be_simulated == 'none':
    parameters['beta'] = 10 * np.ones(n_sim)  # 1 + 3 * np.random.rand(n_sim)
    parameters['p_switch'] = 0.1 * np.ones(n_sim)  # np.random.rand(n_sim)
    parameters['p_reward'] = 0.4 * np.ones(n_sim)  # 0.5 + 0.5 * np.random.rand(n_sim)
    parameters['persev'] = 0 * np.random.rand(n_sim)  # - 0.15
    parameters['sID'] = range(n_sim)

# Load fitted parameters
else:
    parameter_dir = get_paths(run_on_cluster=False)['fitting results']
    print('Loading {0}{1}.\n'.format(parameter_dir, model_to_be_simulated))

    # model_summary = pd.read_csv(parameter_dir + model_to_be_simulated + 'model_summary.csv', index_col=0)
    # parameters['sID'] = ages['sID']
    # for param_name in param_names:
    #     param_idx = [idx for idx in model_summary.index if param_name + '__' in idx]
    #     parameters[param_name] = model_summary.loc[param_idx[:n_subj], 'mean'].values
    # parameters = pd.DataFrame(np.tile(parameters, (n_sim_per_subj, 1)))
    # parameters = pd.DataFrame(np.array(parameters, dtype=float))
    # parameters.columns = np.append(param_names, ['sID'])

    with open(parameter_dir + model_to_be_simulated + '.pickle', 'rb') as handle:
        data = pickle.load(handle)
        trace = data['trace']
        model = data['model']
        model_summary = data['summary']

    for sim_id in range(n_sim_per_subj):
        rand_idx = np.random.randint(0, trace[param_names[0]].shape[0])
        sample_params = pd.DataFrame(np.array([trace[param_name].squeeze()[rand_idx] for param_name in param_names]).T,
                                     columns=param_names)
        sample_params['sID'] = ages['sID']  # index of ages == PyMC's file order == samples_params' order
        parameters = pd.concat([parameters, sample_params], axis=0)

    if 'persev' not in parameters:
        parameters['persev'] = 0

    print('Model contains {0} participants, ages.csv contains {1}.'
          .format(trace[param_names[0]].shape[1], ages.shape[0]))

parameters['eps'] = 0
parameters['p_noisy'] = 1e-5

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

# Initialize task
task_info_path = get_paths(run_on_cluster=False)['PS task info']
task = Task(task_info_path, n_sim_per_subj * n_subj, parameters['sID'])
LL = np.zeros(n_sim_per_subj * n_subj)

print('Simulating {0} agents ({2} simulations each) on {1} trials.\n'.format(n_subj, n_trials, n_sim_per_subj))
for trial in range(n_trials):

    if verbose:
        print("\tTRIAL {0}".format(trial))
    task.prepare_trial()

    # Get p_right (prob that right box is rewarded) and p_choice (prob to choose right) for trials after first trial
    try:
        lik_cor, lik_inc = get_likelihoods(reward, choice, parameters['p_reward'], parameters['p_noisy'])
        persev_bonus = 2 * choice - 1  # recode as -1 for left and +1 for right
        persev_bonus *= parameters['persev']
        [p_right, p_choice] = post_from_lik(lik_cor, lik_inc, persev_bonus,
                                            p_right,
                                            parameters['p_switch'], parameters['eps'], parameters['beta'], verbose=verbose)
    # For first trial
    except NameError:
        print('Using p=0.5!')
        lik_cor = np.nan
        p_right = 0.5 * np.ones(n_sim)
        p_choice = 0.5 * np.ones(n_sim)

    choice = np.random.binomial(n=1, p=p_choice)
    reward = task.produce_reward(choice)
    LL += np.log(p_choice * choice + (1 - p_choice) * (1 - choice))

    if verbose:
        print("lik_cor:", lik_cor)
        print("p_right:", p_right.round(3))
        print("p_choice:", p_choice.round(3))
        print("Choice:", choice)
        print("Reward:", reward)
        print("LL:", LL.round(3))

    # Store trial data
    ps_right[trial] = p_right
    choices[trial] = choice
    rewards[trial] = reward
    correct_boxes[trial] = task.correct_box
    LLs[trial] = LL

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
    # for param_name in param_names:
    #     subj_data[param_name] = np.array(parameters.loc[i, param_name])[version]

    # Save to disc
    file_name = save_dir + "PSBayes_{0}_{1}.csv".format(int(sID), version)
    # print('Saving file {0}'.format(file_name))
    subj_data.to_csv(file_name)

print("Ran and saved {0} simulations ({1} * {2}) to {3}!".
      format(len(parameters['sID']), n_subj, n_sim_per_subj, file_name))
