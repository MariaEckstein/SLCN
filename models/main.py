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
import pandas as pd
from fit_parameters import FitParameters
from minimizer_heatmap import PlotMinimizerHeatmap
from main_helper_functions import *
import os
import re


def interactive_game(sets):

    # Get agent and task stuff
    agent_stuff = get_agent_stuff(sets['data_set'], sets['learning_style'], sets['fit_par_names'])
    task_stuff = get_task_stuff(sets['data_set'])
    comp_stuff = get_comp_stuff(sets['data_set'])

    # Get RL model parameters
    parameters = get_parameter_stuff(sets['data_set'], sets['fit_par_names'])
    gen_pars = get_random_pars(parameters, sets['set_specific_parameters'], sets['learning_style'])

    print('\tAGENT CHARACTERISTICS:\nParameters: {0}'.format(np.round(gen_pars, 2)))
    fit_params = FitParameters(parameters, task_stuff, comp_stuff, agent_stuff)
    fit_params.simulate_agent(sets['data_set'], all_pars=gen_pars, interactive=True)


def simulate(sets, agent_id, save_path):

    # Get info about RL model parameters and minimizer
    parameters = get_parameter_stuff(sets['data_set'], sets['fit_par_names'])

    # Get info about agent and task
    agent_stuff = get_agent_stuff(sets['data_set'], sets['learning_style'], sets['fit_par_names'])
    agent_stuff['id'] = agent_id
    task_stuff = get_task_stuff(sets['data_set'])
    comp_stuff = get_comp_stuff(sets['data_set'])

    # Initialize FitParameters object
    fit_params = FitParameters(parameters, task_stuff, comp_stuff, agent_stuff)

    # Simulate agent based on random parameters
    gen_pars = get_random_pars(parameters, sets['set_specific_parameters'], sets['learning_style'])
    print('Simulating agent {0} with parameters {1}'.format(agent_stuff['id'], np.round(gen_pars, 3)))
    agent_data = fit_params.simulate_agent(sets['data_set'], all_pars=gen_pars)

    # Save data to specified path
    created_file_name = save_path + sets['data_set'] + sets['learning_style'] + str(agent_stuff['id']) + ".csv"
    print("Saving simulated data to {0}".format(created_file_name))
    agent_data.to_csv(created_file_name)


def fit(sets, file_name, fitted_data_path, heatmap_data_path):

    # Get info about RL model parameters and minimizer
    parameters = get_parameter_stuff(sets['data_set'], sets['fit_par_names'])
    minimizer_stuff = get_minimizer_stuff(sets['run_on_cluster'], heatmap_data_path)

    # Get info about agent and task
    agent_stuff = get_agent_stuff(sets['data_set'], sets['learning_style'], sets['fit_par_names'])
    agent_stuff['id'] = int(re.findall('\d+', file_name)[0])
    task_stuff = get_task_stuff(sets['data_set'])
    comp_stuff = get_comp_stuff(sets['data_set'])

    # Read in data
    print("Loading file {0}".format(file_name))
    agent_data = pd.read_csv(file_name)
    agent_data['sID'] = agent_stuff['id']  # redundant but crashes otherwise...

    # Clean data
    if sets['data_set'] == 'Aliens':
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
    rec_pars = fit_params.get_optimal_pars(agent_data, minimizer_stuff,
                                           heatmap_data_path + sets['data_set'] + sets['learning_style'] + str(agent_stuff['id']))
    print("Fitted parameters: {0}".format(np.round(rec_pars, 3)))

    # Calculate NLL,... of minimizing parameters
    agent_data = fit_params.calculate_NLL(vary_pars=rec_pars[np.argwhere(parameters['fit_pars'])],
                                          agent_data=agent_data,
                                          goal='add_decisions_and_fit',
                                          suff='_rec')

    # Write agent_data as csv
    created_file_name = fitted_data_path + sets['data_set'] + sets['learning_style'] + str(agent_stuff['id']) + ".csv"
    print("Saving fitted data to {0}".format(created_file_name))
    agent_data.to_csv(created_file_name)

def plot_heatmaps(sets, agent_id, heatmap_data_path, heatmap_plot_path):

    identifier = sets['data_set'] + sets['learning_style'] + str(agent_id)
    parameters = get_parameter_stuff(sets['data_set'], sets['fit_par_names'])
    plot_heatmap = PlotMinimizerHeatmap(heatmap_data_path + identifier, heatmap_plot_path + identifier)
    plot_heatmap.plot_3d(parameters['fit_par_names'])


def simulate_based_on_data(sets, file_name, simulation_data_path):

    data_set = sets['data_set']
    parameters = get_parameter_stuff(data_set, sets['fit_par_names'])
    agent_stuff = get_agent_stuff(data_set, sets['learning_style'], sets['fit_par_names'])
    fit_params = FitParameters(parameters,
                               get_task_stuff(data_set),
                               get_comp_stuff(data_set),
                               get_agent_stuff(data_set, sets['learning_style'], parameters['fit_par_names']))
    rec_par_columns = [par + '_rec' for par in parameters['par_names']]
    sim_pars = pd.read_csv(file_name).loc[0, rec_par_columns].tolist()
    sim_pars[np.argwhere([par == 'beta' for par in parameters['par_names']])] /= agent_stuff['beta_scaler']
    sim_pars[np.argwhere([par == 'beta_high' for par in parameters['par_names']])] /= agent_stuff['beta_high_scaler']
    # sim_pars[np.argwhere([par == 'TS_bias' for par in parameters['par_names']])] /= agent_stuff['TS_bias_scaler']

    agent_id = pd.read_csv(file_name).loc[0, 'sID']
    for sim_agent_id in range(sets['n_agents']):
        print('Simulating agent {0} with parameters recovered from participant {1} {2}'.format(
            sim_agent_id, agent_id, np.round(sim_pars, 3)))
        agent_data = fit_params.simulate_agent(data_set, sim_pars)

        created_file_name = simulation_data_path + data_set + str(agent_id) + '_' + str(sim_agent_id) + ".csv"
        print("Saving data to {0}".format(created_file_name))
        agent_data.to_csv(created_file_name)


def get_paths(use_humans, data_set, run_on_cluster):

    # Cluster or laptop?
    paths = dict()
    if run_on_cluster:
        paths['base_path'] = '/home/bunge/maria/Desktop/'
    else:
        paths['base_path'] = 'C:/Users/maria/MEGAsync/SLCN/'

    # Check that path exists
    assert os.path.isdir(paths['base_path']), 'No files are found because the specified path {0} does not exist!'.\
        format(paths['base_path'])

    # humans or agents?
    if use_humans:
        paths['agent_data_path'] = paths['base_path'] + data_set + 'humanData/'
    else:
        paths['agent_data_path'] = paths['base_path'] + data_set + 'GenRec/'

    paths['fitted_data_path'] = paths['agent_data_path'] + '/fit_par/'
    paths['heatmap_data_path'] = paths['agent_data_path'] + '/heatmap_data/'
    paths['heatmap_plot_path'] = paths['heatmap_data_path'] + '/plots/'
    paths['simulation_data_path'] = paths['fitted_data_path'] + '/simulations/'

    # Create output folders
    if not os.path.isdir(paths['agent_data_path']):
        os.makedirs(paths['agent_data_path'])
    if not os.path.isdir(paths['fitted_data_path']):
        os.makedirs(paths['fitted_data_path'])
    if not os.path.isdir(paths['heatmap_data_path']):
        os.makedirs(paths['heatmap_data_path'])
    if not os.path.isdir(paths['heatmap_plot_path']):
        os.makedirs(paths['heatmap_plot_path'])
    if not os.path.isdir(paths['simulation_data_path']):
        os.makedirs(paths['simulation_data_path'])

    return paths


def check_user_settings(sets):

    allowed_values = {'run_on_cluster': [True, False],
                      'data_set': ['PS', 'Aliens'],
                      'learning_style': ['s-flat', 'flat', 'simple_flat', 'counter_flat', 'hierarchical', 'Bayes'],
                      'set_specific_parameters': [True, False],
                      'use_humans': [True, False]}

    # Check that every setting is in the allowed range
    for parameter in allowed_values.keys():
        assert sets[parameter] in allowed_values[parameter], 'Variable "{0}" must be one of {1}.' \
            .format(parameter, allowed_values[parameter])
