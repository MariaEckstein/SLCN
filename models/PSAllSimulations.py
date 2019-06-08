import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
sns.set(style='whitegrid')

from PStask import Task
from shared_modeling_simulation import get_paths, update_Q_sim, get_n_trials_back, p_from_Q_sim, get_WSLS_Qs, get_WSLSS_Qs, get_likelihoods, post_from_lik


def get_parameters(data_dir, file_name, n_subj='all', n_sim_per_subj=2):

    orig_parameters = pd.read_csv(data_dir + file_name)
    print("Loaded parameters for model {2} from {0}. parameters.head():\n{1}".format(data_dir, orig_parameters.head(), model_name))

    # Adjust n_subj
    if n_subj == 'all':
        n_subj = len(orig_parameters['sID'])
    parameters = np.tile(orig_parameters[:n_subj], (n_sim_per_subj, 1))  # Same parameters for each simulation of a subject
    parameters = pd.DataFrame(parameters)
    parameters.columns = orig_parameters.columns

    return n_subj, parameters


def simulate_model_from_parameters(parameters, data_dir, n_subj, n_trials=128, n_sim_per_subj=2, verbose=False, make_plots=False):

    """Simulate agents for the model specified in file_name, with the parameters indicated in the referenced file."""

    # GET MODEL PARAMETERS
    n_sim = n_sim_per_subj * n_subj

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
                persev_bonus = 2 * choice - 1  # recode as -1 for choice==0 (left) and +1 for choice==1 (right)
                scaled_persev_bonus = persev_bonus * parameters['persev']  # TODO make sure it has the right shape

                lik_cor, lik_inc = get_likelihoods(rewards[trial - 1], choices[trial - 1],
                                                   parameters['p_reward'], parameters['p_noisy'])

                p_right_, p_right = post_from_lik(lik_cor, lik_inc,
                                                  scaled_persev_bonus, p_right,
                                                  parameters['p_switch'], parameters['beta'])

        # Select an action based on probabilities
        if model_name == 'B':
            choice = np.random.binomial(n=1, p=p_right_)  # don't use persev, don't use softmax
        else:
            choice = np.random.binomial(n=1, p=p_right)  # produces "1" with p_right, and "0" with (1 - p_right)

        # Obtain reward
        reward = task.produce_reward(choice, replace_rewards=False)

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
        param_names = [col for col in parameters.columns if 'sID' not in col]
        for param_name in param_names:
            subj_data[param_name] = np.array(parameters.loc[i, param_name])

        # Save to disc
        file_name = save_dir + "PS_{2}_sim_{0}_{1}.csv".format(int(sID), simID, model_name)
        subj_data.to_csv(file_name)

    print("Ran and saved {0} simulations ({1} * {2}) to {3}!".
          format(len(parameters['sID']), n_subj, n_sim_per_subj, file_name))

    return -np.sum(LL)


def get_quantile_groups(params_ages, col):

    qcol = col + '_quant'
    params_ages[qcol] = np.nan

    # Set adult quantile to 2
    params_ages.loc[params_ages['PreciseYrs'] > 20, qcol] = 2

    # Determine quantiles, separately for both genders
    for gender in ['Male', 'Female']:

        # Get 3 quantiles
        cut_off_values = np.nanquantile(
            params_ages.loc[(params_ages.PreciseYrs < 20) & (params_ages.Gender == gender), col],
            [0, 1 / 3, 2 / 3])
        for cut_off_value, quantile in zip(cut_off_values, np.round([1 / 3, 2 / 3, 1], 2)):
            params_ages.loc[
                (params_ages.PreciseYrs < 20) & (params_ages[col] >= cut_off_value) & (params_ages.Gender == gender),
                qcol] = quantile

    return params_ages


def plot_parameters_against_age_calculate_corr(parameters, ages_file_dir='C:/Users/maria/MEGAsync/SLCNdata/SLCNinfo2.csv'):

    # Load ages dataframe
    ages = pd.read_csv(ages_file_dir)
    ages = ages.rename(columns={'ID': 'sID'})

    # Add age etc. to parameters
    params_ages = parameters.merge(ages)
    params_ages.loc[params_ages['Gender'] == 1, 'Gender'] = 'Female'
    params_ages.loc[params_ages['Gender'] == 2, 'Gender'] = 'Male'

    # # Plot raw parameters against raw age etc.
    # sns.pairplot(params_ages, hue='Gender',
    #              x_vars=['PDS', 'BMI', 'PreciseYrs', 'T1'], y_vars=list(parameters.columns),
    #              plot_kws=dict(size=1))
    # print("Saving plot_params_ages_{1} to {0}".format(data_dir, model_name))
    # plt.savefig("{0}plot_params_ages_{1}.png".format(data_dir, model_name))

    # Same, but with quantile groups for age etc.
    cols = ['PDS', 'BMI', 'PreciseYrs', 'T1']
    param_names = parameters.columns
    plt.figure(figsize=(4 * len(cols), 3 * len(param_names)))
    # fig, axes = plt.subplots(len(cols), len(param_names),
    #                          figsize=(4 * len(cols), 4 * len(param_names)),
    #                          sharex="col", sharey="row",
    #                          squeeze=False)

    i = 0
    for param_name in param_names:
        for col in cols:
            params_ages = get_quantile_groups(params_ages, col)

            # plt.figure()
            # plt.scatter(params_ages[col], params_ages[col + '_quant'], c=params_ages['Gender'])  # check that it worked
            # plt.show()

            plt.subplot(len(param_names), len(cols), i + 1)
            # sns.barplot(x=col + '_quant', y=param_name, hue='Gender', data=params_ages)  # , ax=axes[i])
            sns.lineplot(x=col + '_quant', y=param_name, hue='Gender', data=params_ages, legend=False)  # , ax=axes[i])
            plt.xlabel(col)
            i += 1

    plt.tight_layout()
    plt.savefig("{0}plot_params_ages_quant_lines_{1}.png".format(data_dir, model_name))

    # Get correlations between parameters and age etc.
    corrs = pd.DataFrame()
    for param_name in param_names:
        for col in cols:
            for gen in ('Male', 'Female'):

                # clean_idx = ~np.logical_or(np.isnan(params_ages[col]), np.isnan(params_ages[param_name]))
                clean_idx = 1 - np.isnan(params_ages[col]) | np.isnan(params_ages[param_name])
                gen_idx = params_ages['Gender'] == gen

                corr, p = stats.pearsonr(
                    params_ages.loc[clean_idx & gen_idx, param_name], params_ages.loc[clean_idx & gen_idx, col])

                clean_idx_young = clean_idx & (params_ages['PreciseYrs'] < 20) & gen_idx
                corr_young, p_young = stats.pearsonr(
                    params_ages.loc[clean_idx_young, param_name], params_ages.loc[clean_idx_young, col])

                corrs = corrs.append([[param_name, col, gen, corr, p, corr_young, p_young]])

    corrs.columns = ['param_name', 'charact', 'gender', 'r', 'p', 'r_young', 'p_young']
    corrs.to_csv("{0}corrs_{1}.csv".format(data_dir, model_name), index=False)


# Run simulations for all fitted models
data_dir = 'C:/Users/maria/MEGAsync/SLCN/PShumanData/fitting/map_indiv/simulate_without_replace_rewards/'
file_names = [f for f in os.listdir(data_dir) if '.csv' in f if 'params' in f]
nlls = pd.DataFrame()

for file_name in file_names:
    model_name = [part for part in file_name.split('_') if 'RL' in part or 'B' in part or 'WSLS' in part][0]
    n_subj, parameters = get_parameters(data_dir, file_name)

    # Simulate agents
    nll = simulate_model_from_parameters(parameters, data_dir, n_subj, make_plots=False)
    nlls = nlls.append([[model_name, nll]])

    # Plot parameters against age etc., calculate correlations between parameters and age etc.
    plot_parameters_against_age_calculate_corr(parameters)

nlls.columns = ['model_name', 'simulated_nll']
nlls.to_csv(data_dir + 'nlls.csv', index=False)

# # Debug individual models
# file_name = '2019_06_03/params_RLabcn_2019_6_3_3_7_humans_n_samples100.csv'
# model_name = 'RLabcn'
# n_subj, parameters = get_parameters(data_dir, file_name, n_subj=2)
# simulate_model_from_parameters(parameters, data_dir, n_subj=n_subj, verbose=True)
