import os
import numpy as np
import pandas as pd

from PStask import Task
from shared_modeling_simulation import get_paths, update_Q_sim, get_n_trials_back, p_from_Q_sim, get_WSLS_Qs, get_WSLSS_Qs, get_likelihoods, post_from_lik


# TODO Make plots for all models (early RL ones are wrong)
# TODO Save simulated NLLs (same table as fitted NLL?)


def simulate_model_from_filename(file_name, data_dir, n_trials=128, n_subj='all', n_sim_per_subj=2, verbose=False):

    """Simulate agents for the model specified in file_name, with the parameters indicated in the referenced file."""

    # GET MODEL PARAMETERS
    model_name = [part for part in file_name.split('_') if 'RL' in part or 'B' in part or 'WSLS' in part][0]
    orig_parameters = pd.read_csv(data_dir + file_name)
    # orig_parameters['persev'] += 0.1  # TODO remove after debug!!!
    print("Loaded parameters for model {2} from {0}. parameters.head():\n{1}".format(data_dir, orig_parameters.head(), model_name))

    # Adjust n_subj
    if n_subj == 'all':
        n_subj = len(orig_parameters['sID'])
    n_sim = n_sim_per_subj * n_subj
    parameters = np.tile(orig_parameters[:n_subj], (n_sim_per_subj, 1))  # Same parameters for each simulation of a subject
    parameters = pd.DataFrame(parameters)
    parameters.columns = orig_parameters.columns

    # SIMULATE AGENTS ON THE TASK
    # Get n_trials_back
    n_trials_back = get_n_trials_back(model_name)

    # Get WSLS strategies
    if 'WSLSS' in model_name:
        Qs = get_WSLSS_Qs(2, n_sim)[0]  # only need one trial because I'm not iterating over Qs like in theano
    elif 'WSLS' in model_name:
        Qs = get_WSLS_Qs(2, n_sim)[0]

    # Get reward versions
    reward_versions = pd.read_csv(get_paths(run_on_cluster=False)['PS reward versions'], index_col=0)
    assert np.all((reward_versions["sID"] % 4) == (reward_versions["rewardversion"]))

    # Prepare saving
    rewards = np.zeros((n_trials, n_sim), dtype=int)
    choices = rewards.copy()  # human data: left choice==0; right choice==1 (verified in R script 18/11/17)
    correct_boxes = rewards.copy()  # human data: left choice==0; right choice==1 (verified in R script 18/11/17)
    ps_right = rewards.copy()
    LLs = rewards.copy()
    LL = np.zeros(n_sim)
    if n_trials_back == 0:
        Qs_trials = np.zeros((n_trials, n_sim, 2))
    elif n_trials_back == 1:
        Qs_trials = np.zeros((n_trials, n_sim, 2, 2, 2))
    elif n_trials_back == 2:
        Qs_trials = np.zeros((n_trials, n_sim, 2, 2, 2, 2, 2))

    # Initialize task
    task_info_path = get_paths(run_on_cluster=False)['PS task info']
    task = Task(task_info_path, n_sim)

    print('\nSimulating {0} {3} agents ({2} simulations each) on {1} trials (n_trials_back = {4}).\n'.format(
        n_subj, n_trials, n_sim_per_subj, model_name, n_trials_back))
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
                    n_sim, n_trials_back, verbose=False)

        # Translate Q-values into action probabilities
        if 'RL' in model_name or 'WSLS' in model_name:
            # persev_bonus = np.zeros(n_sim)  # walk-around for avoiding perseverance
            if trial < 2:  # need 2 trials' info to access the right p_right
                p_right = 0.5 * np.ones(n_sim)
            else:
                p_right = p_from_Q_sim(
                    Qs,
                    choices[trial - 2], rewards[trial - 2],
                    choices[trial - 1], rewards[trial - 1],
                    p_right, n_sim,
                    np.array(parameters['beta']), parameters['persev'],
                    n_trials_back, verbose=True)

        elif 'B' in model_name:
            if trial == 0:
                scaled_persev_bonus = np.zeros(n_sim)
                p_right = 0.5 * np.ones(n_sim)
            else:
                persev_bonus = 2 * choice - 1  # recode as -1 for choice==0 (left) and +1 for choice==1 (right)
                scaled_persev_bonus = persev_bonus * parameters['persev']  # TODO make sure it has the right shape

                lik_cor, lik_inc = get_likelihoods(rewards[trial - 1], choices[trial - 1],
                                                   parameters['p_reward'], parameters['p_noisy'])

                p_right_, p_right = post_from_lik(lik_cor, lik_inc,
                                                  scaled_persev_bonus, p_right,
                                                  parameters['p_switch'], parameters['beta'])

        # Select an action based on probabilities
        choice = np.random.binomial(n=1, p=p_right)  # produces "1" with p_right, and "0" with (1 - p_right)

        # Obtain reward
        reward = task.produce_reward(choice)

        # Update log-likelihood
        LL += np.log(p_right * choice + (1 - p_right) * (1 - choice))

        if verbose:
            if 'RL' in model_name:
                print("Qs:\n", np.round(Qs, 2))
            print("p_right:", np.round(p_right, 2))
            print("Choice:", choice)
            print("Reward:", reward)
            print("LL:", LL)

        # Store trial data
        if 'RL' in model_name:
            Qs_trials[trial] = Qs
        ps_right[trial] = p_right
        choices[trial] = choice
        rewards[trial] = reward
        correct_boxes[trial] = task.correct_box
        LLs[trial] = LL

    print("Final LL: {0}".format(np.sum(LL)))

    # SAVE DATA
    # Get save_dir
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
        subj_data["LL"] = LLs[:, i]
        # subj_data["Q_left"] = Qs_left[:, i]
        # subj_data["Q_right"] = Qs_right[:, i]
        param_names = [col for col in parameters.columns if 'sID' not in col]
        for param_name in param_names:
            subj_data[param_name] = np.array(parameters.loc[i, param_name])

        # Save to disc
        file_name = save_dir + "PS_{2}_sim_{0}_{1}.csv".format(int(sID), simID, model_name)
        subj_data.to_csv(file_name)

    print("Ran and saved {0} simulations ({1} * {2}) to {3}!".
          format(len(parameters['sID']), n_subj, n_sim_per_subj, file_name))

    return model_name, -np.sum(LL)


# Run simulations for all fitted models# file_name = 'params_RLab_2019_6_1_12_46_humans_n_samples100.csv'
# file_name = 'params_RLabc_2019_6_1_12_46_humans_n_samples100.csv'
# file_name = 'params_RLabcn_2019_6_1_12_48_humans_n_samples100.csv'
# file_name = 'params_RLabcnnc_2019_6_1_12_48_humans_n_samples100.csv'
# file_name = 'params_RLabcnncS_2019_6_1_12_53_humans_n_samples100.csv'
# file_name = 'params_RLabcnncSi_2019_6_1_13_12_humans_n_samples100.csv'
# file_name = 'params_RLabcnncSS_2019_6_1_13_1_humans_n_samples100.csv'
# file_name = 'params_RLabcnncSSi_2019_6_1_13_20_humans_n_samples100.csv'
# file_name = 'params_WSLS_2019_6_1_12_45_humans_n_samples100.csv'
# file_name = 'params_WSLSS_2019_6_1_12_46_humans_n_samples100.csv'
# file_name = 'params_Bbsr_2019_6_1_12_44_humans_n_samples100.csv'
# file_name = 'params_Bbpsr_2019_6_1_12_45_humans_n_samples100.csv'

data_dir = 'C:/Users/maria/MEGAsync/SLCN/PShumanData/fitting/map_indiv/'
file_names = [f for f in os.listdir(data_dir) if '.csv' in f if 'params' in f]
nlls = pd.DataFrame()
for file_name in file_names[:2]:
    model_name, nll = simulate_model_from_filename(file_name, data_dir)
    nlls = nlls.append([[model_name, nll]])

nlls.columns = ['model_name', 'simulated_nll']
nlls.to_csv(data_dir + 'nlls.csv', index=False)

# file_name = '2019_06_02/params_RLabci_2019_6_1_13_5_humans_n_samples100.csv'  # TODO remove after debug
# simulate_model_from_filename(file_name, data_dir, n_subj=1, verbose=True)
