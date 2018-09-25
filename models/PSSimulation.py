import pickle
import os

import pandas as pd

from shared_modeling_simulation import *
from PStask import Task

# Switches for this script
verbose = False
n_trials = 128  # humans: 128
max_n_subj = 227  # must be > 1  # TODO figure out if 227 or 231
learning_style = 'RL'  # 'Bayes' or 'RL'
# model_to_be_simulated = 'abn_2018_8_14_23_36_humans_n_samples200RL'
model_to_be_simulated = 'none'

# Get save path
save_dir = get_paths(False)['simulations']
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Type in some parameters
if model_to_be_simulated == 'none':
    n_subj = max_n_subj
    alpha = np.random.rand(n_subj)
    eps = 0 * np.ones(n_subj)
    beta = 1 + 3 * np.random.rand(n_subj)
    nalpha = np.random.rand(n_subj)
    calpha_sc = 0 * np.ones(n_subj)
    calpha = alpha * calpha_sc
    cnalpha_sc = 0 * np.ones(n_subj)
    cnalpha = nalpha *cnalpha_sc

# Load fitted parameters
else:
    parameter_dir = get_paths(run_on_cluster=False)['fitting results']
    print('Loading {0}{1}...\n'.format(parameter_dir, model_to_be_simulated))
    with open(parameter_dir + model_to_be_simulated + '.pickle', 'rb') as handle:
        data = pickle.load(handle)
        model_summary = data['summary']
        model = data['model']

    beta_idx = [idx for idx in model_summary.index if 'beta' in idx and '_mu' not in idx]
    beta = model_summary.loc[beta_idx[:max_n_subj], 'mean'].values

    n_subj = len(beta)
    eps = np.zeros(n_subj)

    # Get individual parameters
    if learning_style == 'RL':

        alpha_idx = [idx for idx in model_summary.index if
                     'alpha' in idx and '_mu' not in idx and 'c' not in idx and 'n' not in idx]
        alpha = model_summary.loc[alpha_idx[:n_subj], 'mean'].values

        calpha_idx = [idx for idx in model_summary.index if
                      'calpha' in idx and '_mu' not in idx and 'n' not in idx and 'sc' not in idx]
        calpha = model_summary.loc[calpha_idx[:n_subj], 'mean'].values

        nalpha_idx = [idx for idx in model_summary.index if 'nalpha' in idx and '_mu' not in idx and 'c' not in idx]
        nalpha = model_summary.loc[nalpha_idx[:n_subj], 'mean'].values
        if len(nalpha) == 0:
            nalpha = alpha.copy()

        cnalpha_idx = [idx for idx in model_summary.index if 'cnalpha' in idx and '_mu' not in idx]
        cnalpha = model_summary.loc[cnalpha_idx[:n_subj], 'mean'].values
        if len(cnalpha) == 0:
            cnalpha = nalpha.copy()

    elif learning_style == 'Bayes':

        p_switch_idx = [idx for idx in model_summary.index if 'p_switch' in idx and '_mu' not in idx]
        p_switch = model_summary.loc[p_switch_idx[:n_subj], 'mean'].values

        p_reward_idx = [idx for idx in model_summary.index if 'p_reward' in idx and '_mu' not in idx]
        p_reward = model_summary.loc[p_reward_idx[:n_subj], 'mean'].values

        if len(beta) == 0:
            beta = np.full(n_subj, 999)

        p_noisy = 1e-5 * np.ones(n_subj)

if verbose:
    print("Epsilons: {0}".format(eps.round(3)))
    print("Betas: {0}\n".format(beta.round(3)))
    if learning_style == 'RL':
        print("Alphas: {0}".format(alpha.round(3)))
        print("Calphas: {0}".format(calpha.round(3)))
    elif learning_style == 'Bayes':
        print("p_switch: {0}".format(p_switch.round(3)))
        print("p_reward: {0}:".format(p_reward.round(3)))

# Set up data frames
rewards = np.zeros((n_trials, n_subj))
choices = np.zeros(rewards.shape)
correct_boxes = np.zeros(rewards.shape)
ps_right = np.zeros(rewards.shape)
LLs = np.zeros(rewards.shape)

if learning_style == 'RL':
    Qs_left = np.zeros(rewards.shape)
    Qs_right = np.zeros(rewards.shape)

# Initialize task
task_info_path = get_paths(run_on_cluster=False)['PS task info']
task = Task(task_info_path, n_subj)
LL = np.zeros(n_subj)

print('Simulating {0} agents on {1} trials.\n'.format(n_subj, n_trials))
for trial in range(n_trials):

    if verbose:
        print("\tTRIAL {0}".format(trial))
    task.prepare_trial()

    if learning_style == 'RL':

        # Translate Q-values into action probabilities, make a choice, obtain reward, update Q-values
        try:
            Q_left, Q_right = update_Q(reward, choice, Q_left, Q_right, alpha, nalpha, calpha, cnalpha)
        except NameError:
            print('Using p=0.5!')
            Q_left, Q_right = 0.5 * np.ones(n_subj), 0.5 * np.ones(n_subj)

        p_right = p_from_Q(Q_left, Q_right, beta, eps)
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

    elif learning_style == 'Bayes':

        try:
            lik_cor, lik_inc = get_likelihoods(reward, choice, p_reward, p_noisy)
            p_right = post_from_lik(lik_cor, lik_inc, p_right, p_switch, eps, beta, verbose=verbose)
        except NameError:  # if p_right has not been defined yet
            print('Using p=0.5!')
            p_right = 0.5 * np.ones(n_subj)

        choice = np.random.binomial(n=1, p=p_right)
        reward = task.produce_reward(choice)
        LL += np.log(p_right * choice + (1 - p_right) * (1 - choice))

        if verbose:
            print("p_right:", p_right.round(3))
            print("Choice:", choice)
            print("Reward:", reward)
            print("LL:", LL.round(3))

    # Store trial data
    ps_right[trial] = p_right
    choices[trial] = choice
    rewards[trial] = reward
    correct_boxes[trial] = task.correct_box
    LLs[trial] = LL
    if learning_style == 'RL':
        Qs_left[trial] = Q_left
        Qs_right[trial] = Q_right

# Save data
for sID in range(n_subj):

    # Create pandas DataFrame
    subj_data = pd.DataFrame()
    subj_data["selected_box"] = choices[:, sID]
    subj_data["reward"] = rewards[:, sID]
    subj_data["correct_box"] = correct_boxes[:, sID]
    subj_data["p_right"] = ps_right[:, sID]
    subj_data["sID"] = sID
    subj_data["learning_style"] = learning_style
    subj_data["LL"] = LLs[:, sID]
    subj_data["beta"], subj_data["epsilon"] = beta[sID], eps[sID]
    if learning_style == 'RL':
        subj_data["Q_left"] = Qs_left[:, sID]
        subj_data["Q_right"] = Qs_right[:, sID]
        subj_data["alpha"], subj_data["calpha"], subj_data["nalpha"], subj_data["cnalpha"] = alpha[sID], calpha[sID], nalpha[sID], cnalpha[sID]
    elif learning_style == 'Bayes':
        subj_data['p_switch'], subj_data['p_reward'], subj_data["p_noisy"] = p_switch[sID], p_reward[sID], p_noisy[sID]

    # Save to disc
    file_name = save_dir + "PS" + learning_style + str(sID) + ".csv"
    print('Saving file {0}'.format(file_name))
    subj_data.to_csv(file_name)
