from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import pandas as pd

from AlienTask import Task
from competition_phase import CompetitionPhase
from shared_aliens import alien_initial_Q, update_Qs_sim, softmax, get_alien_paths,\
    simulate_competition_phase, simulate_rainbow_phase, get_summary_rainbow, read_in_human_data

plot_dir = get_alien_paths(False)['fitting results'] + '/SummariesInsteadOfFitting/'


# Switches for this script
def do_simulate(make_plots=False):
    model_name = "hier"
    verbose = False
    n_subj = 31
    n_sim_per_subj = 1
    n_sim = n_subj * n_sim_per_subj
    start_id = 0
    param_names = np.array(['alpha', 'beta', 'forget', 'alpha_high', 'beta_high', 'forget_high'])
    fake_data = False
    human_data_path = get_alien_paths()["human data prepr"]  # "C:/Users/maria/MEGAsync/Berkeley/TaskSets/Data/version3.1/",  # note: human data prepr works for analyzing human behavior, the direct path works just for simulating agents
    model_to_be_simulated = "best_hierarchical"  # "MSE" "MCMC" "specify" "best_hierarchical
    # model_name = "/AliensMSEFitting/18-10-14/f_['alpha' 'beta' 'forget']_[[ 1 10  1]]_2018_10_14_9_47"  # 'Aliens/max_abf_2018_10_10_18_7_humans_n_samples10'  #

    # Get save path
    save_dir = get_alien_paths(False)['simulations']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Get parameters
    parameters = pd.DataFrame(columns=np.append(param_names, ['sID']))
    parameter_dir = get_alien_paths(run_on_cluster=False)['fitting results']

    if model_to_be_simulated == 'specify':

        parameters['alpha'] = 0.1 + 0 * np.random.rand(n_subj)  # 0 < alpha < 0.2
        parameters['beta'] = 1.5 + 0 * np.random.rand(n_subj)  # 1 < beta < 2
        parameters['forget'] = 0.01 + 0 * np.random.rand(n_subj)  # 0 < forget < 0.1

        parameters['alpha_high'] = 0.1 + 0 * np.random.rand(n_subj)  # 0 < alpha_high < 0.2
        parameters['beta_high'] = 1.5 + 0 * np.random.rand(n_subj)  # 1 < beta < 2
        parameters['forget_high'] = 0.01 + 0 * np.random.rand(n_subj)

        parameters['sID'] = range(n_subj)
        parameters['total_NLL'] = np.nan

    # Load fitted parameters
    elif model_to_be_simulated == 'best_hierarchical':
        read_dir = parameter_dir + '/SummariesInsteadOfFitting/selected_agents4_good.csv'
        print('Using parameters in {} for simulation!'.format(read_dir))
        all_parameters = pd.read_csv(read_dir, usecols=param_names)  # pd.read_csv(plot_dir + 'ag_summary_for_paper.csv', index_col=0).loc[param_names]
        # Same parameters for all subj
        parameters = all_parameters.loc[0].values.reshape((1, len(param_names))) * np.ones((n_subj, len(param_names)))
        parameters = pd.DataFrame(parameters, columns=all_parameters.columns.values)
        # # Different parameters for each subj
        # parameters = all_parameters.loc[:(n_subj-1)]
        parameters.loc[:, 'sID'] = range(n_subj)

        stop = 4

    elif model_to_be_simulated == 'MSE':

        read_dir = parameter_dir + '/AliensMSEFitting/ten_best_indiv_{}.csv'.format(model_name)
        print("Reading in {}".format(read_dir))
        parameters = pd.read_csv(read_dir, index_col=0)
        n_subj = min(n_subj, parameters.shape[0])
        parameters = parameters[:n_subj]
        parameters['sID'] = range(n_subj)

        if 'alpha_high' not in parameters:
            parameters['alpha_high'] = parameters['alpha'].copy()
        if 'beta_high' not in parameters:
            parameters['beta_high'] = parameters['beta'].copy()
        if 'forget_high' not in parameters:
            parameters['forget_high'] = parameters['forget'].copy()

    elif model_to_be_simulated == 'MCMC':

        print('Loading {0}{1}.\n'.format(parameter_dir, model_name))
        with open(parameter_dir + model_name + '.pickle', 'rb') as handle:
            data = pickle.load(handle)
            model_summary = data['summary']
            model = data['model']

        model_summary = pd.read_csv(parameter_dir + model_name + '_summary.csv', index_col=0)
        for param_name in param_names:
            param_idx = [idx for idx in model_summary.index if param_name + '__' in idx]
            parameters[param_name] = model_summary.loc[param_idx[:n_subj], 'mean'].values
        parameters['sID'] = range(n_subj)
        parameters = pd.DataFrame(np.tile(parameters, (n_sim_per_subj, 1)))
        parameters = pd.DataFrame(np.array(parameters, dtype=float))
        parameters.columns = np.append(param_names, ['sID'])

    if verbose:
        print("Parameters: {}".format(parameters.round(3)))

    # Parameter shapes
    beta_shape = (n_sim, 1)  # Q_low_sub.shape -> [n_subj, n_actions]
    beta_high_shape = (n_sim, 1)  # Q_high_sub.shape -> [n_subj, n_TS]
    forget_high_shape = (n_sim, 1, 1)  # -> [n_subj, n_seasons, n_TS]
    forget_shape = (n_sim, 1, 1, 1)  # Q_low[0].shape -> [n_subj, n_TS, n_aliens, n_actions]

    # Get numbers of things
    n_seasons, n_aliens, n_actions = 3, 4, 3
    n_TS = n_seasons

    # Initialize task
    task = Task(n_subj)
    n_trials, _, _ = task.get_trial_sequence(human_data_path,
                                             n_subj, n_sim_per_subj, range(n_subj), fake_data,
                                             phases=['1InitialLearning', '2CloudySeason'])  #
    print("n_trials", n_trials)

    n_trials_ = {'1InitialLearn': np.sum(task.phase=='1InitialLearning'),
                 '2CloudySeason': np.sum(task.phase=='2CloudySeason'),
                 '4RainbowSeason': 4 * n_aliens}
    trials = {'1InitialLearn': range(n_trials_['1InitialLearn']),
              '2CloudySeason': range(n_trials_['1InitialLearn'], n_trials_['1InitialLearn'] + n_trials_['2CloudySeason']),
              '4RainbowSeason': range(n_trials_['2CloudySeason'], n_trials_['2CloudySeason'] + n_trials_['4RainbowSeason'])}

    # For saving data
    seasons = np.zeros([n_trials, n_sim], dtype=int)
    TSs = np.zeros([n_trials, n_sim], dtype=int)
    aliens = np.zeros([n_trials, n_sim], dtype=int)
    actions = np.zeros([n_trials, n_sim], dtype=int)
    rewards = np.zeros([n_trials, n_sim])
    corrects = np.zeros([n_trials, n_sim])
    p_lows = np.zeros([n_trials, n_sim, n_actions])
    Q_lows = np.zeros([n_trials, n_sim, n_TS, n_aliens, n_actions])
    Q_highs = np.zeros([n_trials, n_sim, n_seasons, n_TS])
    phase = np.zeros(seasons.shape)

    print('Simulating {0} {2} agents on {1} trials.\n'.format(n_sim, n_trials, model_name))
    # print('Parameters:\n{}'.format(parameters[np.append(param_names, ['total_NLL'])].round(2)))

    Q_low = alien_initial_Q * np.ones([n_sim, n_TS, n_aliens, n_actions])
    Q_high = alien_initial_Q * np.ones([n_sim, n_seasons, n_TS])

    # Bring parameters into the right shape
    alpha = parameters['alpha'].values
    beta = parameters['beta'].values.reshape(beta_shape)
    forget = parameters['forget'].values.reshape(forget_shape)
    alpha_high = parameters['alpha_high'].values
    beta_high = parameters['beta_high'].values.reshape(beta_high_shape)
    forget_high = parameters['forget_high'].values.reshape(forget_high_shape)

    # InitialLearning phase
    print("Working on Initial Learning!")
    for trial in trials['1InitialLearn']:

        # Observe stimuli
        season, alien = task.present_stimulus(trial)

        # Respond and update Q-values
        [Q_low, Q_high, TS, action, correct, reward, p_low] =\
            update_Qs_sim(season, alien,
                          Q_low, Q_high,
                          beta, beta_high, alpha, alpha_high, forget, forget_high,
                          n_sim, n_actions, n_TS, task, verbose=verbose)

        # Store trial data
        seasons[trial] = season
        phase[trial] = 1
        TSs[trial] = TS
        aliens[trial] = alien
        actions[trial] = action
        rewards[trial] = reward
        corrects[trial] = correct
        Q_highs[trial] = Q_high
        Q_lows[trial] = Q_low
        p_lows[trial] = p_low

    # Save final Q-values to simulate subsequent phases
    final_Q_low = Q_low.copy()
    final_Q_high = Q_high.copy()
    # first_cloudy_trial = trial + 1

    # Cloudy season
    print("Working on Cloudy Season!")
    for trial in trials['2CloudySeason']:

        # Observe stimuli
        old_season = season.copy()
        season, alien = task.present_stimulus(trial)

        # Take care of season switches
        if trial == list(trials['2CloudySeason'])[0]:
            season_switches = np.ones(n_sim, dtype=bool)
        else:
            season_switches = season != old_season

        if model_name == 'hier':
            Q_high[season_switches] = alien_initial_Q  # re-start search for the right TS when season changes
        elif model_name == 'flat':
            Q_low[season_switches] = alien_initial_Q  # re-learn a new policy from scratch when season changes
        else:
            raise(NameError, 'model_name must be either "flat" or "hier"!')

        # Print info
        if verbose:
            print("\n\tTRIAL {0}".format(trial))
            print("season switches:", season_switches)
            print("season:", season)
            print("alien:", alien)

        # Update Q-values
        [Q_low, Q_high, TS, action, correct, reward, p_low] =\
            update_Qs_sim(0 * season, alien,
                          Q_low, Q_high,
                          beta, beta_high, alpha, alpha_high, forget, forget_high,
                          n_sim, n_actions, n_TS, task, verbose=verbose)

        # Store trial data
        seasons[trial] = season
        phase[trial] = 2
        TSs[trial] = TS
        aliens[trial] = alien
        actions[trial] = action
        rewards[trial] = reward
        corrects[trial] = correct
        Q_lows[trial] = Q_low
        Q_highs[trial] = Q_high
        p_lows[trial] = p_low

    # Read in human data
    n_hum, hum_aliens, hum_seasons, hum_corrects, hum_actions, hum_rewards, hum_rainbow_dat, hum_comp_dat = read_in_human_data(human_data_path, n_trials, n_aliens, n_actions)

    assert np.all(hum_seasons == seasons)  # only works when simulating exactly 31 agents
    assert np.all(hum_aliens == aliens)

    # Plot learning curves
    if make_plots:
        plt.figure()
        for i, phase in enumerate(['1InitialLearn', '2CloudySeason']):
            learning_curve = np.mean(corrects[trials[phase]], axis=1)
            hum_learning_curve = np.mean(hum_corrects[trials[phase]], axis=1)
            plt.subplot(2, 1, i+1)
            plt.title(phase)
            plt.plot(trials[phase], learning_curve, label='Simulation')
            plt.plot(trials[phase], hum_learning_curve, label='Humans')
            plt.legend()
            plt.ylim(0, 1)
            plt.xlabel('Trial')
            plt.ylabel('% correct')
        plt.tight_layout()

        # Plot repetition learning curves
        plt.figure()
        for phase in ['1InitialLearn', '2CloudySeason']:
            season_changes = np.array([seasons[i, 0] != seasons[i+1, 0] for i in list(trials[phase])[:-1]])
            season_changes = np.insert(season_changes, 0, False)
            season_presentation = np.cumsum(season_changes)
            repetition = season_presentation // n_seasons
            n_trials_per_rep = np.sum(repetition==0)
            n_rep_rep = np.sum(season_presentation==0)

            # Prepare simulated data
            corrects_rep = corrects[trials[phase]].reshape((3, n_trials_per_rep, n_sim))
            learning_curve_rep = np.mean(corrects_rep, axis=2)
            rep_rep = learning_curve_rep.reshape((3, 3, n_rep_rep))
            rep_rep = np.mean(rep_rep, axis=1)
            learning = np.mean(rep_rep[-1] - rep_rep[0])  # Average increase from first to last repetition

            # Prepare human data
            hum_corrects_rep = hum_corrects[trials[phase]].reshape((3, n_trials_per_rep, n_hum))
            hum_learning_curve_rep = np.mean(hum_corrects_rep, axis=2)
            hum_rep_rep = hum_learning_curve_rep.reshape((3, 3, n_rep_rep))
            hum_rep_rep = np.mean(hum_rep_rep, axis=1)
            hum_learning = np.mean(hum_rep_rep[-1] - hum_rep_rep[0])  # Average increase from first to last repetition

            plt.figure()
            plt.subplot(2, 2, 1)  # Plot simulations
            plt.title('Simulations')
            for rep in range(3):
                plt.plot(range(n_trials_per_rep), learning_curve_rep[rep], label=rep)
            plt.ylim(0, 1)
            plt.ylabel('% correct ({})'.format(phase))
            plt.xlabel('Trial in repetition')
            plt.legend()

            plt.subplot(2, 2, 2)  # Plot simulations
            plt.title('Simulations')
            for rep in range(3):
                plt.plot(range(n_rep_rep), rep_rep[rep], label=rep)
            plt.ylim(0, 1)
            plt.ylabel('% correct ({})'.format(phase))
            plt.xlabel('Trial in repetition')
            plt.legend()

            plt.subplot(2, 2, 3)  # Plot humans
            plt.title('Humans')
            for rep in range(3):
                plt.plot(range(n_trials_per_rep), hum_learning_curve_rep[rep], label=rep)
            plt.ylim(0, 1)
            plt.ylabel('% correct ({})'.format(phase))
            plt.xlabel('Trial in repetition')
            plt.legend()

            plt.subplot(2, 2, 4)  # Plot simulations
            plt.title('Humans')
            for rep in range(3):
                plt.plot(range(n_rep_rep), hum_rep_rep[rep], label=rep)
            plt.ylim(0, 1)
            plt.ylabel('% correct ({})'.format(phase))
            plt.xlabel('Trial in repetition')
            plt.legend()
            plt.tight_layout()

    # Competition phase
    print("Working on Competition Phase!")
    comp_data = simulate_competition_phase(model_name, final_Q_high, final_Q_low, task,
                                           n_seasons, n_aliens, n_sim, beta_high)

    # Plot competition results
    if make_plots:
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.axhline(y=0.5, color='grey', linestyle='--')
        plt.bar(comp_data['phase'] + comp_data['choice'], comp_data['perc_selected_better'], yerr=comp_data['se'])
        plt.xticks(rotation=15)
        plt.ylim(0, 1)
        plt.ylabel("frac. selected better option")
        plt.subplot(2, 1, 2)
        sum_dat = comp_data.groupby('phase').aggregate('mean')
        plt.axhline(y=0.5, color='grey', linestyle='--')
        plt.bar(sum_dat.index, sum_dat['perc_selected_better'], yerr=sum_dat['se'])
        plt.ylim(0, 1)
        plt.ylabel("frac. selected better option")
        plt.tight_layout()
        plt.show()

    # Rainbow phase
    print("Working on Rainbow Phase!")
    rainbow_dat = simulate_rainbow_phase(n_seasons, model_name, n_sim,
                                         beta, beta_high, final_Q_low, final_Q_high)

    TS_choices = get_summary_rainbow(n_aliens, n_seasons, rainbow_dat, task)
    hum_TS_choices = get_summary_rainbow(n_aliens, n_seasons, hum_rainbow_dat, task)

    # plot rainbow phase (this INCLUDES choices that are correct in multiple TS!)
    # if make_plots:
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     plt.title('Rainbow phase')
    #     ax.bar(np.arange(4)+0.15, np.sum(TS_choices, axis=1), 0.3, label='Simulatinos')
    #     ax.bar(np.arange(4)-0.15, np.sum(hum_TS_choices, axis=1), 0.3, label='Humans')
    #     ax.set_ylabel('TS chosen (count)')
    #     ax.set_xticks(range(4))
    #     ax.set_xticklabels(['TS0', 'TS1', 'TS2', 'noTS'])
    #     ax.legend()
    #     plt.show()

    # Save simulation data
    for sID in range(n_sim):

        agent_ID = sID + start_id
        # Create pandas DataFrame
        subj_data = pd.DataFrame()
        subj_data["context"] = seasons[:, sID]
        # subj_data["phase"] = phase[:, sID]
        subj_data["TS_chosen"] = TSs[:, sID]
        subj_data["sad_alien"] = aliens[:, sID]
        subj_data["item_chosen"] = actions[:, sID]
        subj_data["reward"] = rewards[:, sID]
        subj_data["correct"] = corrects[:, sID]
        subj_data["trial_type"] = "feed-aliens"
        subj_data["trial_index"] = np.arange(n_trials)
        # subj_data["p_low"] = p_norms[:, sID]
        subj_data["sID"] = agent_ID
        subj_data["block.type"] = "normal"
        subj_data["model_name"] = model_name
        for param_name in param_names:
            subj_data[param_name] = np.array(parameters.loc[sID, param_name])
        for season in range(n_seasons):
            for TS in range(n_TS):
                subj_data["Q_high_c{0}TS{1}".format(season, TS)] = Q_highs[:, sID, season, TS]
        for TS in range(n_TS):
            for item in range(n_actions):
                for alien in range(n_aliens):
                    subj_data["Q_low_TS{0}s{1}a{2}".format(TS, alien, item)] = Q_lows[:, sID, TS, alien, item]

        # Save to disc
        file_name = save_dir + "aliens_" + model_name + '_' + str(agent_ID) + ".csv"
        print('Saving file {0}'.format(file_name))
        subj_data.to_csv(file_name)

    return (alpha, beta, forget, alpha_high, beta_high, forget_high), (seasons, TSs, aliens, actions, rewards)


do_simulate(make_plots=False)
