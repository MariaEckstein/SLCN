import numpy as np
import pandas as pd
import scipy.stats as stats

import os
import pickle

from shared_modeling_simulation import Shared
from PStask import Task

# Switches for this script
verbose = False
n_trials = 201
n_subj = 50
learning_style = 'RL'  # 'Bayes' or 'RL'

# Get save path
shared = Shared()
save_dir = shared.get_paths()['simulations']
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Load fitted parameters
# parameter_dir = shared.get_paths()['fitting results']
# with open(parameter_dir + 'trace_2018_7_2_nsubj155Hier_mu.pickle', 'rb') as handle:
#     trace = pickle.load(handle)

# Population-level priors
epsilon_mu, epsilon_sd = 0.1, 0.2
if learning_style == 'RL':
    alpha_mu, alpha_sd = 0.25, 0.1
    calpha_scaler_mu, calpha_scaler_sd = 0.5, 0.5
    beta_mu, beta_sd, beta_upper = 5, 3, 50

elif learning_style == 'Bayes':
    p_switch_mu, p_switch_sd = 0.1, 0.1
    p_reward_mu, p_reward_sd = 0.75, 0.3
    p_noisy_mu, p_noisy_sd = 0.05, 0.1

# Individual parameters (drawn from truncated normal with parameters above)
epsilon = 0.1 * np.ones(n_subj)
# epsilon = stats.truncnorm(-epsilon_mu / epsilon_sd, (1 - epsilon_mu) / epsilon_sd,
#                            loc=epsilon_mu, scale=epsilon_sd).rvs(n_subj)
if learning_style == 'RL':
    alpha = 0.8 * np.ones(n_subj)
    # alpha = stats.truncnorm(-alpha_mu / alpha_sd, (1 - alpha_mu) / alpha_sd,
    #                         loc=alpha_mu, scale=alpha_sd).rvs(n_subj)
    calpha_scaler = 0.8 * np.ones(n_subj)
    # calpha_scaler = stats.truncnorm(-calpha_scaler_mu / calpha_scaler_sd, (1 - calpha_scaler_mu) / calpha_scaler_sd,
    #                         loc=alpha_mu, scale=alpha_sd).rvs(n_subj)
    calpha = alpha * calpha_scaler
    beta = 3 * np.ones(n_subj)
    # beta = stats.truncnorm(-beta_mu / beta_sd, (beta_upper - beta_mu) / beta_sd,
    #                        loc=beta_mu, scale=beta_sd).rvs(n_subj)

elif learning_style == 'Bayes':
    p_switch = stats.truncnorm(-p_switch_mu / p_switch_sd, (1 - p_switch_mu) / p_switch_sd,
                               loc=p_switch_mu, scale=p_switch_sd).rvs(n_subj)
    # p_switch = 0.1 * np.ones(n_subj)
    p_reward = stats.truncnorm(-p_reward_mu / p_reward_sd, (1 - p_reward_mu) / p_reward_sd,
                               loc=p_reward_mu, scale=p_reward_sd).rvs(n_subj)
    # p_reward = 0.75 * np.ones(n_subj)
    p_noisy_task = stats.truncnorm(-p_noisy_mu / p_noisy_sd, (1 - p_noisy_mu) / p_noisy_sd,
                                   loc=p_noisy_mu, scale=p_noisy_sd).rvs(n_subj)
    # p_noisy_task = 0.01 * np.ones(n_subj)

if verbose:
    print("Epsilons: {0}".format(epsilon.round(2)))
    if learning_style == 'RL':
        print("Alphas: {0}".format(alpha.round(2)))
        print("Calphas: {0}".format(calpha.round(2)))
        print("Betas: {0}\n".format(beta.round(2)))
    elif learning_style == 'Bayes':
        print("p_switch: {0}".format(p_switch.round(2)))
        print("p_reward: {0}:".format(p_reward.round(2)))
        print("p_noisy_task: {0}\n".format(p_noisy_task.round(2)))

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
task_info_path = shared.get_paths()['PS task info']
task = Task(task_info_path, n_subj)
LL = np.zeros(n_subj)

print('Simulating {0} agents on {1} trials.\n'.format(n_subj, n_trials))
for trial in range(n_trials):

    task.prepare_trial()

    if learning_style == 'RL':

        # Translate Q-values into action probabilities, make a choice, obtain reward, update Q-values
        try:
            Q_left, Q_right = shared.update_Q(reward, choice, Q_left, Q_right, alpha, calpha)
        except NameError:
            Q_left, Q_right = 0.5 * np.ones(n_subj), 0.5 * np.ones(n_subj)

        p_right = shared.p_from_Q(Q_left, Q_right, beta, epsilon)
        choice = np.random.binomial(n=1, p=p_right)
        reward = task.produce_reward(choice)
        LL += np.log(p_right * choice + (1 - p_right) * (1 - choice))

        if verbose:
            print("\tTRIAL {0}".format(trial))
            print("p_right:", p_right.round(2))
            print("Choice:", choice)
            print("Reward:", reward)
            print("Q_left:", Q_left.round(3))
            print("Q_right:", Q_right.round(3))
            print("LL:", LL)

    elif learning_style == 'Bayes':

        try:
            lik_cor, lik_inc = shared.get_likelihoods(reward, choice, p_reward, p_noisy_task)
            p_right = shared.post_from_lik(lik_cor, lik_inc, p_right, p_switch, epsilon)
        except NameError:
            p_right = 0.5 * np.ones(n_subj)

        choice = np.random.binomial(n=1, p=p_right)
        reward = task.produce_reward(choice)
        LL += np.log(p_right * choice + (1 - p_right) * (1 - choice))

        if verbose:
            print("\tTRIAL {0}".format(trial))
            print("p_right:", p_right.round(3))
            print("Choice:", choice)
            print("Reward:", reward)
            print("LL:", LL)

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
    if learning_style == 'RL':
        subj_data["Q_left"] = Qs_left[:, sID]
        subj_data["Q_right"] = Qs_right[:, sID]
        subj_data["alpha"], subj_data["calpha"], subj_data["beta"] = alpha[sID], calpha[sID], beta[sID]
    elif learning_style == 'Bayes':
        subj_data['p_switch'], subj_data['p_reward'], subj_data['p_noisy_task'], subj_data['p_reward'] = p_switch, p_reward, p_noisy_task, p_reward

    # Save to disc
    file_name = save_dir + "PS" + learning_style + str(sID) + ".csv"
    print('Saving file {0}'.format(file_name))
    subj_data.to_csv(file_name)
