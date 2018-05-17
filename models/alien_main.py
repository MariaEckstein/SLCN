import numpy as np
import glob
import os
import pandas as pd
from parameters import Parameters
from fit_parameters import FitParameters
from gen_rec import GenRec
from visualize_agent import VisualizeAgent


# TDs / bugs
# First trial is currently not recorded (the data files are missing the alien in the first trial)

# What should be done?
interactive_game = False
quick_generate_and_recover = False
fit_human_data = False
simulate_agents = True

# Model fitting parameters
n_iter = 20
n_agents = 200
agent_start_id = 500
base_path = 'C:/Users/maria/MEGAsync/Berkeley/TaskSets'  # CLUSTER: base_path = '/home/bunge/maria/Desktop/Aliens'
data_path = base_path + '/AlienGenRec/'
human_data_path = 'C:/Users/maria/MEGAsync/Berkeley/TaskSets/Data/version3.1'   # CLUSTER: human_data_path = base_path + '/humanData/'
save_fitted_human_path = human_data_path + '/fit_par'
n_simulated_agents_per_participant = 100

# Task parameters
n_actions = 3
                # TS0
TSs = np.array([[[1, 6, 1],  # alien0, items0-2
                 [1, 1, 4],  # alien1, items0-2
                 [5, 1, 1],  # etc.
                 [10, 1, 1]],
                # TS1
               [[1, 1, 2],  # alien0, items0-2
                [1, 8, 1],  # etc.
                [1, 1, 7],
                [1, 3, 1]],
                # TS2
               [[1, 1, 7],  # TS2
                [3, 1, 1],
                [1, 3, 1],
                [2, 1, 1]]])

task_stuff = {'n_trials_per_alien': 13,  # 13
              'n_blocks': 7,  # 7 = 3 (initial learn) + 2 (refr2) + 2 (refr3)
              'n_aliens': 4,
              'n_actions': n_actions,
              'n_contexts': 3,
              'TS': TSs}
comp_stuff = {'phases': ['contexts', 'context-aliens', 'items', 'aliens'],
              'n_blocks': {'contexts': 3, 'context-aliens': 3, 'items': 3, 'aliens': 3}}
agent_stuff = {'name': 'alien',
               'n_TS': 3,
               'learning_style': 'hierarchical',
               'mix_probs': False}

parameters = Parameters(par_names=['alpha', 'beta', 'epsilon', 'forget'],  # Rewards <= 10 means than beta is 10 times as much!
                        fit_pars=np.ones(6, dtype=bool),  # which parameters will be fitted?
                        par_hard_limits=((0., 1.),  (1., 15.), (0., 1.), (0., 1.)),  # no values fitted outside
                        par_soft_limits=((0., 0.5), (1., 6.),  (0., 0.25), (0., 0.1)),  # no simulations outside
                        default_pars_lim=np.array([0.1, 1., 0., 0.]))  # when a parameter is fixed
gen_pars = parameters.default_pars_lim
gen_pars[np.array(parameters.par_names) == 'epsilon'] = 0

# Play the game to test everything
if interactive_game:
    agent_stuff['id'] = agent_start_id

    # Adjust parameters
    for feature_name in ['learning_style', 'mix_probs']:
        feat = input('{0} (leave blank for "{1}"):'.format(feature_name, agent_stuff[feature_name]))
        if feat:
            agent_stuff[feature_name] = feat
    for par_name in parameters.par_names:
        par = input('{0} (leave blank for {1}):'.format(par_name, gen_pars[np.array(parameters.par_names) == par_name]))
        if par:
            gen_pars[np.array(parameters.par_names) == par_name] = par

    print('\tAGENT CHARACTERISTICS:\n'
          'Learning style: {0}\nMix probs: {1}\nParameters: {2}'.format(
          str(agent_stuff['learning_style']), str(agent_stuff['mix_probs']), str(np.round(gen_pars, 2))))
    fit_params = FitParameters(parameters=parameters,
                               task_stuff=task_stuff,
                               comp_stuff=comp_stuff,
                               agent_stuff=agent_stuff)
    agent_data = fit_params.get_agent_data(way='interactive',
                                           all_params_lim=gen_pars)

# Generate and recover
if quick_generate_and_recover:

    # Adjust parameters
    for feature_name in ['learning_style', 'mix_probs']:
        feat = input('{0} (leave blank for "{1}"):'.format(feature_name, agent_stuff[feature_name]))
        if feat:
            agent_stuff[feature_name] = feat

    # Decide which parameters will be fit (others will just be known)
    fit_pars = np.array(parameters.par_names) == 'epsilon'  # np.ones(len(parameters.par_names), dtype=bool)  #
    parameters.set_fit_pars(fit_pars)
    parameters.adjust_fit_pars(agent_stuff['learning_style'])

    fit_par_names = '_'.join([parameters.par_names[int(i)] for i in np.argwhere(parameters.fit_pars)])
    agent_stuff['fit_par'] = fit_par_names

    # Specify where data will be saved
    save_agent_path = data_path
    if not os.path.isdir(save_agent_path):
        os.makedirs(save_agent_path)

    # gen_rec = GenRec(parameters=parameters, full_genrec_path=data_path + '/genrec.csv')
    agent_id = agent_start_id
    while agent_id < agent_start_id + n_agents:
        print("Agent " + str(agent_id))

        agent_stuff['id'] = agent_id
        fit_params = FitParameters(parameters=parameters,
                                   task_stuff=task_stuff,
                                   comp_stuff=comp_stuff,
                                   agent_stuff=agent_stuff)

        # Simulate agent based on random parameters
        gen_pars = parameters.create_random_params(scale='lim', get_all=True, mode='soft')
        # gen_pars[np.array(parameters.par_names) == 'epsilon'] = 0
        print("Generated parameters: " + str(np.round(gen_pars, 3)))
        agent_data = fit_params.get_agent_data(way='simulate',
                                               all_params_lim=gen_pars)
        # Make sure that data are based on the right parameters
        for i, par in enumerate(parameters.par_names):
            assert(gen_pars[i] == agent_data[par][0])

        # Fit the free parameters (remaining parameters will be set to gen_pars)
        rec_pars = fit_params.get_optimal_pars(agent_data=agent_data,
                                               n_iter=n_iter,
                                               default_pars_lim=gen_pars)
        print("Recovered parameters: " + str(np.round(rec_pars, 3)))

        # Calculate model fit and add stuff
        agent_data = fit_params.calculate_NLL(params_inf=parameters.lim_to_inf(rec_pars),
                                              agent_data=agent_data,
                                              default_pars_lim=gen_pars,
                                              goal='add_decisions_and_fit')
        fit_params.write_agent_data(agent_data=agent_data,
                                    save_path=save_agent_path,
                                    file_name='/alien')
        agent_id += 1

# Fit parameters to human data
if fit_human_data:
    fit_pars = np.array([1, 1, 0, 0], dtype=bool)
    parameters.set_fit_pars(fit_pars)
    parameters.adjust_fit_pars(agent_stuff['learning_style'])

    fit_par_names = '_'.join([parameters.par_names[int(i)] for i in np.argwhere(parameters.fit_pars)])
    agent_stuff['fit_par'] = fit_par_names

    file_names = glob.glob(human_data_path + '/aliens*csv')
    for file_name in file_names:
        agent_id = int(file_name.split('aliens')[1].split('.csv')[0])
        print('\tPARTICIPANT {0}'.format(agent_id))
        agent_stuff['id'] = agent_id

        # Specify model
        fit_params = FitParameters(parameters=parameters,
                                   task_stuff=task_stuff,
                                   comp_stuff=comp_stuff,
                                   agent_stuff=agent_stuff)

        # Specify where data will be saved
        if not os.path.isdir(save_fitted_human_path):
            os.makedirs(save_fitted_human_path)

        # Get and clean participant data
        agent_data = fit_params.get_agent_data(way='real_data', data_path=file_name)
        agent_data = agent_data.rename(columns={'TS': 'context'})  # rename "TS" column to "context"
        context_names = [str(i) for i in range(agent_stuff['n_TS'])]
        item_names = range(task_stuff['n_actions'])
        agent_data = agent_data.loc[(agent_data['context'].isin(context_names)) & (agent_data['item_chosen'].isin(item_names))]
        agent_data.index = range(agent_data.shape[0])

        # Fit free parameters
        rec_pars = fit_params.get_optimal_pars(agent_data=agent_data,
                                               n_iter=n_iter)
        print("Recovered parameters: " + str(np.round(rec_pars, 3)))

        # Calculate fit and add stuff
        agent_data = fit_params.calculate_NLL(params_inf=parameters.lim_to_inf(rec_pars),
                                              agent_data=agent_data,
                                              goal='add_decisions_and_fit')
        agent_data['sID'] = agent_id
        fit_params.write_agent_data(agent_data=agent_data,
                                    save_path=save_fitted_human_path,
                                    file_name='alien')

# Simulate specific agents
human_simulation_base_path = 'C:/Users/maria/MEGAsync/Berkeley/TaskSets/AlienGenRec/FlatStimulusAlphaBeta/'
save_fitted_human_path = human_simulation_base_path + 'FittedParticipants/'
save_simulated_human_path = human_simulation_base_path + 'SimulatedParticipants/'
if not os.path.isdir(save_simulated_human_path):
    os.makedirs(save_simulated_human_path)
if simulate_agents:

    # Specify which agents will be simulated
    file_names = glob.glob(save_fitted_human_path + '/*alien*csv')
    for file_name in file_names:
        subj_file = pd.read_csv(file_name)
        agent_stuff['learning_style'] = subj_file['learning_style'][0]
        agent_stuff['mix_probs'] = subj_file['mix_probs'][0]
        gen_pars = np.empty(len(parameters.par_names))
        for i, par_name in enumerate(parameters.par_names):
            gen_pars[i] = subj_file[par_name + '_rec'][0]

        # Simulate a number of n_agents data sets
        agent_id = 0
        while agent_id < n_simulated_agents_per_participant:
            agent_stuff['id'] = agent_id
            fit_params = FitParameters(parameters=parameters,
                                       task_stuff=task_stuff,
                                       comp_stuff=comp_stuff,
                                       agent_stuff=agent_stuff)
            agent_data = fit_params.get_agent_data(way='simulate',
                                                   all_params_lim=gen_pars)
            sim_save_path = '{0}sim_{1}_{2}.csv'.format(
                save_simulated_human_path, str(subj_file['sID'][0]), str(agent_id))
            agent_data.to_csv(sim_save_path)

            agent_id += 1
