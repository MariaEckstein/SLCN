# Things to keep in mind / knobs to turn;
# separate alphas / betas for different phases
# suppress at level of policy (i.e., within softmax Q_to_p)

# TDs / bugs
# Make main a nice class?
# Fix paths in minimizer_heatmap?
# Calculate Q(TS) by hand from Q_low for flat agents in competition phase (alien-same-season)!
# Record the right Q_high in cloudy season in alien_record_data (index 4)!
# Save Q and p for competition phase?

# Import statements
import numpy as np
import os
import pandas as pd
from fit_parameters import FitParameters
from minimizer_heatmap import PlotMinimizerHeatmap


def simulate(agent_id, sets):

    check_user_settings(sets)
    paths = get_paths(sets)

    # Get info about RL model parameters and minimizer
    parameters = get_parameter_stuff(sets['fit_par_names'])

    # Get info about agent and task
    agent_stuff = get_agent_stuff(sets['data_set'], sets['fit_par_names'])
    agent_stuff['id'] = agent_id
    task_stuff = get_task_stuff(sets['data_set'])
    comp_stuff = get_comp_stuff(sets['data_set'])

    # Initialize FitParameters object
    fit_params = FitParameters(parameters, task_stuff, comp_stuff, agent_stuff)

    # Simulate agent based on random parameters
    gen_pars = get_random_par(parameters, sets['set_specific_parameters'], sets['learning_style'])
    print('Simulating agent {0} with parameters {1}'.format(agent_stuff['id'], np.round(gen_pars, 3)))
    agent_data = fit_params.simulate_agent(all_pars=gen_pars)

    # Save data to specified path
    created_file_name = paths['agent_data_path'] + paths['file_name_pattern'] + str(agent_stuff['id']) + ".csv"
    print("Saving simulated data to {0}".format(created_file_name))
    agent_data.to_csv(created_file_name)


def fit(file_name, sets):

    check_user_settings(sets)
    paths = get_paths(sets)

    # Get info about RL model parameters and minimizer
    parameters = get_parameter_stuff(sets['fit_par_names'])
    minimizer_stuff = get_minimizer_stuff(sets['run_on_cluster'], paths['plot_data_path'])

    # Get info about agent and task
    agent_stuff = get_agent_stuff(sets['data_set'], sets['fit_par_names'])
    agent_stuff['id'] = int(file_name.split(paths['file_name_pattern'])[1].split('.csv')[0])
    task_stuff = get_task_stuff(sets['data_set'])
    comp_stuff = get_comp_stuff(sets['data_set'])

    # Read in data
    print("Loading file {0}".format(file_name))
    agent_data = pd.read_csv(file_name)
    agent_data['sID'] = agent_stuff['id']  # redundant but crashes otherwise...

    # Clean data
    if sets['use_humans']:

        # Remove all rows that do not contain 1InitialLearning data (-> jsPysch format)
        agent_data = agent_data.rename(columns={'TS': 'context'})  # rename "TS" column to "context"
        context_names = [str(TS) for TS in range(agent_stuff['n_TS'])]
        item_names = range(task_stuff['n_actions'])
        agent_data = agent_data.loc[
            (agent_data['context'].isin(context_names)) & (agent_data['item_chosen'].isin(item_names))]
    else:

        # Remove all phases that are not InitialLearning
        agent_data = agent_data.loc[agent_data['phase'] == '1InitialLearning']
    agent_data.index = range(agent_data.shape[0])

    # Look up generated parameters and display
    if 'alpha' in agent_data.columns:
        gen_pars = agent_data.loc[0, parameters['par_names']]
        gen_pars[np.argwhere(parameters['par_names'] == 'beta')] /= agent_stuff['beta_scaler']
        gen_pars[np.argwhere(parameters['par_names'] == 'beta_high')] /= agent_stuff['beta_high_scaler']
        print("True parameters: {0}".format(np.round(gen_pars.values.astype(np.double), 3)))

    # Find parameters that minimize NLL
    fit_params = FitParameters(parameters, task_stuff, comp_stuff, agent_stuff)
    rec_pars = fit_params.get_optimal_pars(agent_data=agent_data, minimizer_stuff=minimizer_stuff)
    print("Fitted parameters: {0}".format(np.round(rec_pars, 3)))

    # Calculate NLL,... of minimizing parameters
    agent_data = fit_params.calculate_NLL(vary_pars=rec_pars[np.argwhere(parameters['fit_pars'])],
                                          agent_data=agent_data,
                                          goal='add_decisions_and_fit',
                                          suff='_rec')

    # Write agent_data as csv
    created_file_name = paths['fitting_save_path'] + paths['file_name_pattern'] + str(agent_stuff['id']) + ".csv"
    print("Saving fitted data to {0}".format(created_file_name))
    agent_data.to_csv(created_file_name)

# def simulate_based_on_data(sets):
#
#     if sets['use_humans'] and sets['simulate_agents_post_fit']:
#         for sim_agent_id in range(sets['n_agents']):
#             print('Simulating agent {0} with parameters recovered from participant {2} {1}'.format(
#                 sim_agent_id, np.round(rec_pars, 3), agent_stuff['id']))
#             agent_data = fit_params.simulate_agent(all_pars=rec_pars)
#
#             created_file_name = paths['fitting_save_path'] + paths['file_name_pattern'] + str(agent_stuff['id']) + '_' + str(sim_agent_id) + ".csv"
#             print("Saving data to {0}".format(created_file_name))
#             agent_data.to_csv(created_file_name)


def plot_heatmaps(agent_id, sets):

    paths = get_paths(sets)
    parameters = get_parameter_stuff(sets['fit_par_names'])

    data_path = paths['plot_data_path'] + '/ID' + str(agent_id)
    plot_heatmap = PlotMinimizerHeatmap(data_path)
    plot_heatmap.plot_3d(parameters['fit_par_names'], paths['plot_save_path'])


def interactive_game(sets):

    # Get agent and task stuff
    agent_stuff = get_agent_stuff(sets['data_set'], sets['fit_par_names'])
    agent_stuff['id'] = 0
    task_stuff = get_task_stuff(sets['data_set'])
    comp_stuff = get_comp_stuff(sets['data_set'])

    # Get RL model parameters
    parameters = get_parameter_stuff(sets['fit_par_names'])
    gen_pars = get_random_par(parameters, True, sets['learning_style'])

    print('\tAGENT CHARACTERISTICS:\nParameters: {0}'.format(np.round(gen_pars, 2)))
    fit_params = FitParameters(parameters, task_stuff, comp_stuff, agent_stuff)
    fit_params.simulate_agent(all_pars=gen_pars, interactive=True)


def check_user_settings(settings_dictionary):

    # Check if all user-specified settings are allowed; raise error if not
    sets = settings_dictionary
    allowed_values = {'run_on_cluster': [True, False],
                      'data_set': ['PS', 'Aliens'],
                      'learning_style': ['s-flat', 'flat', 'hierarchical', 'Bayes'],
                      'use_humans': [True, False],
                      'set_specific_parameters': [True, False]}
    for parameter in allowed_values.keys():
        assert sets[parameter] in allowed_values[parameter], 'Variable "{0}" must be one of {1}.' \
            .format(parameter, allowed_values[parameter])


def get_paths(sets):

    # Get paths on cluster
    paths = dict()
    if sets['run_on_cluster']:
        paths['base_path'] = '/home/bunge/maria/Desktop/' + sets['data_set']
        paths['human_data_path'] = paths['base_path'] + '/humanData/'

    # Get paths on local computer
    else:
        if sets['data_set'] == 'Aliens':
            paths['base_path'] = 'C:/Users/maria/MEGAsync/Berkeley/TaskSets'
            paths['human_data_path'] = paths['base_path'] + '/Data/version3.1/'
        else:
            paths['base_path'] = 'C:/Users/maria/MEGAsync/SLCN'
            paths['human_data_path'] = paths['base_path'] + '/data/PSResults'

    # Define output folders
    paths['agent_data_path'] = paths['base_path'] + '/GenRec/'
    paths['plot_data_path'] = paths['agent_data_path'] + '/heatmap_data/'
    paths['plot_save_path'] = paths['plot_data_path'] + '/plots/'

    if sets['use_humans']:
        paths['fitting_save_path'] = paths['human_data_path'] + '/fit_pars/'
        paths['file_name_pattern'] = sets['data_set']
    else:
        paths['fitting_save_path'] = paths['agent_data_path'] + '/fit_pars/'
        paths['file_name_pattern'] = 'sim_'

    # Create output folders
    if not os.path.isdir(paths['agent_data_path']):
        os.makedirs(paths['agent_data_path'])
    if not os.path.isdir(paths['plot_data_path']):
        os.makedirs(paths['plot_data_path'])
    if not os.path.isdir(paths['plot_save_path']):
        os.makedirs(paths['plot_save_path'])
    if not os.path.isdir(paths['fitting_save_path']):
        os.makedirs(paths['fitting_save_path'])

    return paths


def get_task_stuff(data_set):

    if data_set == 'PS':
        task_stuff = {'n_actions': 2,
                      'p_reward': 0.75,
                      'n_trials': 200,
                      'av_run_length': 10}  # DOUBLE-CHECK!!!!

    else:
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

        task_stuff = {'phases': ['1InitialLearning', '2CloudySeason', 'Refresher2', '3PickAliens',
                                 'Refresher3', '5RainbowSeason', 'Mixed'],
                      'n_trials_per_alien': np.array([13, 10, 7, np.nan, 7, 1, 7]),
                      'n_blocks': np.array([3, 3, 2, np.nan, 2, 3, 3]),
                      'n_aliens': 4,
                      'n_actions': 3,
                      'n_contexts': 3,
                      'TS': TSs}

    return task_stuff


def get_agent_stuff(data_set, fit_par_names):
    return {'name': data_set,
            'n_TS': 3,
            'fit_par': '_'.join(fit_par_names),
            'beta_scaler': 2,
            'beta_high_scaler': 4,
            'TS_bias_scaler': 6}


def get_parameter_stuff(fit_par_names):

    par_names = ['alpha', 'alpha_high', 'beta', 'beta_high', 'epsilon', 'forget', 'TS_bias']
    default_pars = np.array([.1,  99,       .5,       99,       0.,        0.,       .3])
    par_hard_limits = ((0., 1.), (0., 1.), (0., 1.), (0., 1.), (0., 1.),  (0., 1.), (0., 1.))
    par_soft_limits = ((0., .5), (0., .5), (0., 1.), (0., 1.), (0., .25), (0., .1), (0., 1.))

    return {'par_hard_limits': par_hard_limits,  # no fitting outside
            'par_soft_limits': par_soft_limits,  # no simulations outside
            'default_pars': default_pars,
            'par_names': par_names,
            'fit_par_names': fit_par_names,
            'fit_pars': np.array([par in fit_par_names for par in par_names])}


def get_comp_stuff(data_set):
    if data_set == 'Aliens':
        return {'phases': ['season', 'alien-same-season', 'item', 'alien'],
                'n_blocks': {'season': 3, 'alien-same-season': 3, 'item': 3, 'alien': 3}}
    else:
        return None


def get_minimizer_stuff(run_on_cluster, plot_data_path):

    if run_on_cluster:
        return {'save_plot_data': False,
                'plot_data_path': plot_data_path,
                'verbose': False,
                'brute_Ns': 50,
                'hoppin_T': 10.0,
                'hoppin_stepsize': 0.5,
                'NM_niter': 300,
                'NM_xatol': .01,
                'NM_fatol': 1e-6,
                'NM_maxfev': 1000}

    else:
        return {'save_plot_data': True,
                'plot_data_path': plot_data_path,
                'verbose': True,
                'brute_Ns': 4,
                'hoppin_T': 10.0,
                'hoppin_stepsize': 0.5,
                'NM_niter': 3,
                'NM_xatol': .1,
                'NM_fatol': .1,
                'NM_maxfev': 10}


def get_random_par(parameters, set_specific_parameters, learning_style):

    # Get random parameters within soft limits, with default pars for those that should not be fit
    gen_pars = np.array([lim[0] + np.random.rand() * (lim[1] - lim[0]) for lim in parameters['par_soft_limits']])
    fixed_par_idx = np.invert(parameters['fit_pars'])
    gen_pars[fixed_par_idx] = parameters['default_pars'][fixed_par_idx]

    # Adjust beta_high and TS_bias if learning style is flat
    beta_high_TS_bias = np.argwhere([par in ['beta_high', 'TS_bias'] for par in parameters['par_names']])
    if learning_style == 'flat':
        gen_pars[beta_high_TS_bias] = [100., 100.]
    elif learning_style == 's-flat':
        gen_pars[beta_high_TS_bias] = [0., 1.]

    # Alternatively, let user set parameters
    if set_specific_parameters:
        for i, par in enumerate(gen_pars):
            change_par = input('Accept {0} of {1}? If not, type a new number.'.
                               format(parameters['par_names'][i], np.round(par, 2)))
            if change_par:
                gen_pars[i] = float(change_par)

    return gen_pars
