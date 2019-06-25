import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
sns.set(style='whitegrid')

from PStask import Task
from shared_modeling_simulation import get_paths, update_Q_sim, get_n_trials_back, p_from_Q_sim, get_WSLS_Qs, get_WSLSS_Qs, get_likelihoods, post_from_lik
from modeling_helpers import load_data


def get_parameters(data_dir, file_name, n_subj='all', n_sim_per_subj=2):

    orig_parameters = pd.read_csv(data_dir + file_name)
    print("Loaded parameters for model {2} from {0}\nParameters.head():\n{1}".format(data_dir, orig_parameters.head(), model_name))

    # Adjust n_subj
    if n_subj == 'all':
        n_subj = len(orig_parameters['sID'])
    parameters = np.tile(orig_parameters[:n_subj], (n_sim_per_subj, 1))  # Same parameters for each simulation of a subject
    parameters = pd.DataFrame(parameters)
    parameters.columns = orig_parameters.columns

    return n_subj, parameters


def simulate_model_from_parameters(parameters, data_dir, n_subj, n_trials=120, n_sim_per_subj=2, verbose=False, make_plots=False, calculate_NLL_from_data=False, loaded_data=None):

    """Simulate agents for the model specified in file_name, with the parameters indicated in the referenced file."""

    # # TODO remove after debugging!!!
    # model_name = 'RLabcnp'
    # parameters['beta'] = 3.63
    # parameters['persev'] = -0.006
    # parameters['alpha'] = 0.999
    # parameters['nalpha'] = 0.304
    # parameters['calpha'] = 0.999 * parameters['alpha']
    # parameters['cnalpha'] = 0.999 * parameters['nalpha']

    # GET MODEL PARAMETERS
    n_sim = n_sim_per_subj * n_subj

    # SIMULATE AGENTS ON THE TASK
    # Get n_trials_back
    n_trials_back = get_n_trials_back(model_name)

    # Get reward versions
    reward_versions = pd.read_csv(get_paths(run_on_cluster=False)['PS reward versions'], index_col=0)
    assert np.all((reward_versions["sID"] % 4) == (reward_versions["rewardversion"]))

    # Prepare saving
    if not calculate_NLL_from_data:
        rewards = np.zeros((n_trials, n_sim), dtype=int)
        choices = rewards.copy()  # human data: left choice==0; right choice==1 (verified in R script 18/11/17)
        correct_boxes = rewards.copy()  # human data: left choice==0; right choice==1 (verified in R script 18/11/17)
    else:
        rewards, choices = loaded_data['rewards'], loaded_data['choices']
    ps_right = np.zeros((n_trials, n_sim))

    # Get WSLS strategies
    if 'WSLSS' in model_name:
        Qs = get_WSLSS_Qs(2, n_sim)[0]  # only need one trial because I'm not iterating over Qs like in theano
    elif 'WSLS' in model_name:
        Qs = get_WSLS_Qs(2, n_sim)[0]

    if n_trials_back == 0:
        Qs_trials = np.zeros((n_trials, n_sim, 2))
    elif n_trials_back == 1:
        Qs_trials = np.zeros((n_trials, n_sim, 2, 2, 2))
    elif n_trials_back == 2:
        Qs_trials = np.zeros((n_trials, n_sim, 2, 2, 2, 2, 2))

    # Initialize task
    task_info_path = get_paths(run_on_cluster=False)['PS task info']
    task = Task(task_info_path, n_sim)

    if not calculate_NLL_from_data:
        print('\nSimulating {0} {3} agents ({2} simulations each) on {1} trials (n_trials_back = {4}).\n'.format(
            n_subj, n_trials, n_sim_per_subj, model_name, n_trials_back))
    else:
        print('\nCalculating NLLs for {0} {2} agents on {1} trials (n_trials_back = {3}).'.format(
            n_subj, n_trials, model_name, n_trials_back))

    for trial in range(n_trials):

        if verbose:
            print("\n\tTRIAL {0}".format(trial))
        task.prepare_trial()

        # Initialize Q-values
        if 'RL' in model_name:
            if trial <= 2:
                if n_trials_back == 0:
                    Qs = 0.5 * np.ones((n_sim, 2))  # shape: (n_sim, n_choice)
                elif n_trials_back == 1:
                    if 'i' in model_name:
                        Qs = get_WSLS_Qs(2, n_sim)[0]
                    else:
                        Qs = 0.5 * np.ones((n_sim, 2, 2, 2))  # shape: (n_sim, n_prev_choice, n_prev_reward, n_choice)
                elif n_trials_back == 2:
                    if 'i' in model_name:
                        Qs = get_WSLSS_Qs(2, n_sim)[0]
                    else:
                        Qs = 0.5 * np.ones((n_sim, 2, 2, 2, 2, 2))  # shape: (n_sim, n_prev_prev_choice, n_prev_prev_reward, n_prev_choice, n_prev_reward, n_choice)
                _ = 0  # for theano.scan debug issues

            # Update Q-values (starting on third trial)
            else:
                Qs, _ = update_Q_sim(
                    choices[trial - 3], rewards[trial - 3],
                    choices[trial - 2], rewards[trial - 2],  # prev_prev_choice, prev_prev_reward -> used for defining the state
                    choices[trial - 1], rewards[trial - 1],  # prev_choice, prev_reward -> used for updating Q-values
                    Qs, _,
                    parameters['alpha'], parameters['nalpha'], parameters['calpha'], parameters['cnalpha'],
                    n_sim, n_trials_back, verbose=verbose)

        # Translate Q-values into action probabilities
        if 'RL' in model_name or 'WSLS' in model_name:
            if trial < 2:  # need 2 trials' info to access the right p_right
                p_right = 0.5 * np.ones(n_sim)  # Simulations are using their initialized values asap
            else:
                p_right = p_from_Q_sim(
                    Qs,
                    choices[trial - 2], rewards[trial - 2],
                    choices[trial - 1], rewards[trial - 1],
                    p_right, n_sim,
                    np.array(parameters['beta']), parameters['persev'],
                    n_trials_back, verbose=verbose)

        elif 'B' in model_name:
            if trial == 0:
                scaled_persev_bonus = np.zeros(n_sim)
                p_right = 0.5 * np.ones(n_sim)
                if model_name == 'B':
                    p_right_ = 0.5 * np.ones(n_sim)
            else:
                if not calculate_NLL_from_data:
                    persev_bonus = 2 * choice - 1  # recode as -1 for choice==0 (left) and +1 for choice==1 (right)
                else:
                    persev_bonus = 2 * choices[trial-1] - 1
                scaled_persev_bonus = persev_bonus * parameters['persev']  # TODO make sure it has the right shape

                lik_cor, lik_inc = get_likelihoods(rewards[trial - 1], choices[trial - 1],
                                                   parameters['p_reward'], parameters['p_noisy'])

                p_right_, p_right = post_from_lik(lik_cor, lik_inc,
                                                  scaled_persev_bonus, p_right,
                                                  parameters['p_switch'], parameters['beta'])

        # Select an action based on probabilities
        if not calculate_NLL_from_data:
            if model_name == 'B':
                choice = np.random.binomial(n=1, p=p_right_)  # don't use persev, don't use softmax
            else:
                choice = np.random.binomial(n=1, p=p_right)  # produces "1" with p_right, and "0" with (1 - p_right)

            # Obtain reward
            reward = task.produce_reward(choice, replace_rewards=False)

        if verbose:
            if 'RL' in model_name:
                print("Qs:\n", np.round(Qs, 2))
            print("p_right:", np.round(p_right, 2))
            if not calculate_NLL_from_data:
                print("Choice:", choice)
                print("Reward:", reward)

        # Store trial data
        if 'RL' in model_name:
            Qs_trials[trial] = Qs
        if model_name == 'B':
            ps_right[trial] = p_right_
        else:
            ps_right[trial] = p_right
        if not calculate_NLL_from_data:
            choices[trial] = choice
            rewards[trial] = reward
            correct_boxes[trial] = task.correct_box
    trialwise_LLs = np.log(ps_right * choices + (1 - ps_right) * (1 - choices))[3:]  # remove 3 trials like in pymc3
    subjwise_LLs = np.sum(trialwise_LLs, axis=0)

    print("Final LL: {0}".format(np.sum(subjwise_LLs)))

    # Plot development of Q-values
    if 'RL' in model_name and make_plots:
        for subj in range(min(n_sim, 5)):
            plt.figure()
            if n_trials_back == 0:
                plt.plot(Qs_trials[:, subj, 0], label='L')
                plt.plot(Qs_trials[:, subj, 1], label='R')

            elif n_trials_back == 1:
                # what I should see:
                # LOL & R0R overlap perfectly (also L1L & R1R, L0R & R0L, and L1R & R1L)
                # L0L & L0R trade off (also L1L & L1R, etc.)
                plt.plot(Qs_trials[:, subj, 0, 0, 0], label='L0L')
                plt.plot(Qs_trials[:, subj, 1, 0, 1], label='R0R')
                plt.plot(Qs_trials[:, subj, 0, 0, 1], label='L0R')
                plt.plot(Qs_trials[:, subj, 1, 0, 0], label='R0L')
                plt.plot(Qs_trials[:, subj, 1, 1, 1], label='R1R')
                plt.plot(Qs_trials[:, subj, 0, 1, 0], label='L1L')
                plt.plot(Qs_trials[:, subj, 0, 1, 1], label='L1R')
                plt.plot(Qs_trials[:, subj, 1, 1, 0], label='R1L')
            plt.ylim((0, 1))
            plt.legend()
        plt.show()

    # SAVE DATA
    # Get save_dir
    if not calculate_NLL_from_data:
        save_dir = data_dir + 'simulations/' + model_name + '/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save individual csv for each subject (same format as humans)
        for i, sID in enumerate(parameters['sID']):

            simID = int(np.floor(i / n_subj))

            # Create pandas DataFrame
            subj_data = pd.DataFrame()
            subj_data["selected_box"] = choices[:, i]
            subj_data["reward"] = rewards[:, i]
            subj_data["correct_box"] = correct_boxes[:, i]
            subj_data["p_right"] = ps_right[:, i]
            subj_data["sID"] = sID
            subj_data["simID"] = simID
            subj_data["LL"] = subjwise_LLs[:, i]
            param_names = [col for col in parameters.columns if 'sID' not in col]
            for param_name in param_names:
                subj_data[param_name] = np.array(parameters.loc[i, param_name])

            # Save to disc
            file_name = save_dir + "PS_{2}_sim_{0}_{1}.csv".format(int(sID), simID, model_name)
            subj_data.to_csv(file_name)

        print("Ran and saved {0} simulations ({1} * {2}) to {3}!".
              format(len(parameters['sID']), n_subj, n_sim_per_subj, file_name))

    else:
        print("Calculated {0} NLLs for {1} subjects and saved everything to {2}!".
              format(len(parameters['sID']), n_subj, data_dir))
        trialwise_LLs = pd.DataFrame(trialwise_LLs[:, :n_subj], columns=parameters['sID'][:n_subj])  # remove double subjects
        trialwise_LLs.to_csv(data_dir + 'plots/trialwise_LLs_' + model_name + '_sim.csv', index=False)
        subjwise_LLs = pd.DataFrame(subjwise_LLs[:n_subj], index=parameters['sID'][:n_subj], columns=['NLL_sim'])
        subjwise_LLs.to_csv(data_dir + 'plots/subjwise_LLs_' + model_name + '_sim.csv')

    return -np.sum(subjwise_LLs).values[0]  # total NLL of the model


# Run simulations for all fitted models
calculate_NLL_from_data = True
n_sim_per_subj = 2
verbose = False
data_dir = 'C:/Users/maria/MEGAsync/SLCN/PShumanData/fitting/map_indiv/new_map_models4/'
file_names = [f for f in os.listdir(data_dir) if ('.csv' in f) and ('params' in f) and not ('params_g' in f)]
print("Found {0} models to simulate / calculate NLLs.".format(len(file_names)))
nlls = pd.DataFrame()

if calculate_NLL_from_data:
    n_subj, rewards, choices, group, n_groups, age = load_data(False, n_groups=1, n_subj='all',
                                                               kids_and_teens_only=False,
                                                               n_trials=120,
                                                               fit_slopes=False)
    loaded_data = {'rewards': np.array(np.tile(rewards, 2), dtype=int),
                   'choices': np.array(np.tile(choices, 2), dtype=int)}
    indiv_nlls = pd.DataFrame()
else:
    loaded_data = None

for file_name in file_names:
    model_name = [part for part in file_name.split('_') if ('RL' in part) or ('B' in part) or ('WSLS' in part)][0]
    n_subj, parameters = get_parameters(data_dir, file_name)

    if calculate_NLL_from_data:
        assert np.all(age['sID'].values[:n_subj] == parameters['sID'].values[
                                                :n_subj]), "Parameters don't fit loaded data!! Results will be wrong!!"

    # Simulate agents / calculate NLLs
    nll = simulate_model_from_parameters(parameters, data_dir, n_subj, make_plots=False, calculate_NLL_from_data=calculate_NLL_from_data, verbose=verbose, loaded_data=loaded_data, n_sim_per_subj=n_sim_per_subj)
    nlls = nlls.append([[model_name, nll]])

if calculate_NLL_from_data:
    nlls.columns = ['model_name', 'fitted_nll']
else:
    nlls.columns = ['model_name', 'simulated_nll']

print("Saving nlls to {0}.".format(data_dir))
nlls.to_csv(data_dir + 'plots/modelwise_LLs_sim.csv', index=False)

# # Debug individual models
# file_name = '2019_06_03/params_RLabcn_2019_6_3_3_7_humans_n_samples100.csv'
# model_name = 'RLabcn'
# n_subj, parameters = get_parameters(data_dir, file_name, n_subj=2)
# simulate_model_from_parameters(parameters, data_dir, n_subj=n_subj, verbose=True)
