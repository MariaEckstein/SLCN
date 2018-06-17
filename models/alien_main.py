import numpy as np
import glob
import os
import pandas as pd
from parameters import Parameters
from fit_parameters import FitParameters


# Things to keep in mind / knobs to turn;
# separate alphas / betas for low level & level
# separate alphas / betas for different phases
# suppress at level of policy (i.e., within softmax Q_to_p)

# TDs / bugs
# Calculate Q(TS) by hand from Q_low for flat agents in competition phase (alien-same-season)!
# Record the right Q_high in cloudy season in alien_record_data (index 4)!
# Save Q and p for competition phase?
# Fix context and self.context!

# Model fitting parameters
n_agents = 10
agent_start_id = 800
base_path = 'C:/Users/maria/MEGAsync/Berkeley/TaskSets'  # CLUSTER: base_path = '/home/bunge/maria/Desktop/Aliens'
data_path = base_path + '/AlienGenRec/MaxInsteadOfSoftmax/'
human_data_path = 'C:/Users/maria/MEGAsync/Berkeley/TaskSets/Data/version3.1/'   # CLUSTER: existing_data_path = base_path + '/humanData/'
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

task_stuff = {'phases': ['1InitialLearning', '2CloudySeason', 'Refresher2', '3PickAliens',
                         'Refresher3', '5RainbowSeason', 'Mixed'],
              'n_trials_per_alien': np.array([13, 10, 7, np.nan, 7, 1, 7]),  # np.array([1, 1, 1, np.nan, 1, 1, 1]), #
              'n_blocks': np.array([3, 3, 2, np.nan, 2, 3, 3]),  # np.array([1, 1, 1, np.nan, 1, 1, 1]), #
              'n_aliens': 4,
              'n_actions': n_actions,
              'n_contexts': 3,
              'TS': TSs}
comp_stuff = {'phases': ['season', 'alien-same-season', 'item', 'alien'],
              'n_blocks': {'season': 3, 'alien-same-season': 3, 'item': 3, 'alien': 3}}
agent_stuff = {'name': 'alien',
               'n_TS': 3,
               'learning_style': 'hierarchical',
               'mix_probs': False}

parameters = {'par_names': ['alpha', 'alpha_high', 'beta', 'beta_high', 'epsilon', 'forget',
                            'create_TS_biased_prefer_new', 'create_TS_biased_copy_old'],
              'par_hard_limits': ((0., 1.), (0., 1.), (0., 1.), (0., 1.), (0., 1.),  (0., 1.), (0., 1.), (0., 1.)),  # no values fitted outside; beta will be multiplied by 6 inside of alien_agents.py!
              'par_soft_limits': ((0., .5), (0., .5), (0., 1.), (0., 1.), (0., .25), (0., .1), (0., 1.), (0., 1.)),  # no simulations outside
              'default_pars': np.array([.1, 0.,    .1,       0.,       0.,        0.,       0.,       0.])}  # when a parameter is fixed

# Ask user what should be done
interactive_game = False
use_existing_data = input('Work with existing data (humans / simulations)? Type T for yes.') in ['True', 'T', 'Y']
use_humans = False
if use_existing_data:
    use_humans = input('Work with humans (otherwise will be agents)? Type T for yes.') in ['True', 'T', 'Y']
    if use_humans:
        do_fit_model = input('Fit model parameters to human data? Type T for yes.') in ['True', 'T', 'Y']
        do_simulate_agents = input('Simulate data based on fitted parameters? Type T for yes.') in ['True', 'T', 'Y']
        save_path = human_data_path + '/fit_par/'
        file_name_pattern = 'aliens'
        file_names = glob.glob(human_data_path + file_name_pattern + '*csv')
    else:
        do_fit_model = True
        do_simulate_agents = False
        save_path = data_path + '/fit_par/'
        file_name_pattern = 'sim_'
        file_names = glob.glob(data_path + file_name_pattern + '*csv')
else:
    do_fit_model = input('Fit model parameters after simulation? Type T for yes.') in ['True', 'T', 'Y']
    do_simulate_agents = True
    save_path = data_path
    file_name_pattern = 'sim_'
    file_names = range(agent_start_id, agent_start_id + n_agents)

if do_fit_model:
    n_iter = 20
    n_iter_input = input('How many iterations for brute and basinhopping? Type a number; leave blank for 20.')
    if n_iter_input:
        n_iter = int(n_iter_input)

if use_existing_data:
    get_agent_data_way = 'real_data'
else:
    get_agent_data_way = 'simulate'

fit_pars = np.zeros(len(parameters['par_names']), dtype=bool)
for i, par_name in enumerate(parameters['par_names']):
    fit_par = input('Vary / fit parameter {0}? Type T for yes.'.format(par_name)) in ['True', 'T', 'Y']
    if fit_par:
        fit_pars[i] = fit_par
parameters['fit_pars'] = fit_pars
fit_par_names = '_'.join([parameters['par_names'][int(i)] for i in np.argwhere(parameters['fit_pars'])])
agent_stuff['fit_par'] = fit_par_names

# Create folder to save output files
if not os.path.isdir(save_path):
    os.makedirs(save_path)

# Create / fit each agent / person
for file_name in file_names:

    # Get agent id
    if use_existing_data:
        agent_id = int(file_name.split(file_name_pattern)[1].split('.csv')[0])
    else:
        agent_id = file_name
    print('\tPARTICIPANT {0}'.format(agent_id))
    agent_stuff['id'] = agent_id

    # Specify model
    fit_params = FitParameters(parameters=parameters, task_stuff=task_stuff,
                               comp_stuff=comp_stuff, agent_stuff=agent_stuff)

    # Step 1: Get data
    if use_existing_data:
        # Read in existing data (human or agent)
        agent_data = pd.read_csv(file_name)

    elif do_simulate_agents:
        # Simulate agents from scratch
        gen_pars = [limit[0] + np.random.rand() * (limit[1] - limit[0]) for limit in parameters['par_soft_limits']]
        gen_pars[np.invert(fit_pars)] = 0
        print('Simulating {0} agent {1} with parameters {2}'.format(
            agent_stuff['learning_style'], agent_id, np.round(gen_pars, 2)))
        agent_data = fit_params.simulate_agent(all_pars=gen_pars, all_Q_columns=False)
    agent_data['sID'] = agent_id  # redundant but crashes otherwise...

    # Step 2: Fit parameters
    if do_fit_model:

        # Clean data
        if use_humans:
            # Remove all rows that do not contain 1InitialLearning data (-> jsPysch format)
            agent_data = agent_data.rename(columns={'TS': 'context'})  # rename "TS" column to "context"
            context_names = [str(TS) for TS in range(agent_stuff['n_TS'])]
            item_names = range(task_stuff['n_actions'])
            agent_data = agent_data.loc[(agent_data['context'].isin(context_names)) & (agent_data['item_chosen'].isin(item_names))]
        else:
            # Remove all phases that are not InitialLearning
            agent_data = agent_data.loc[agent_data['phase'] == '1InitialLearning']
        agent_data.index = range(agent_data.shape[0])

        if 'alpha' in agent_data.columns:
            # Look up generated parameters and display
            gen_pars = agent_data.loc[0, parameters['par_names']]
            gen_pars[2] /= 6  # beta
            print("True parameters: " + str(np.round(gen_pars.values.astype(np.double), 3)))

        # Find parameters that minimize function
        rec_pars = fit_params.get_optimal_pars(agent_data=agent_data, n_iter=n_iter)
        print("Fitted parameters: " + str(np.round(rec_pars, 3)))

        # Calculate NLL,... of minimizing parameters
        agent_data = fit_params.calculate_NLL(vary_pars=rec_pars[np.argwhere(parameters['fit_pars'])],
                                              agent_data=agent_data,
                                              goal='add_decisions_and_fit',
                                              suff=['_' + agent_stuff['learning_style'] + '_' + fit_par_names])

    # Write agent_data as csv
    created_file_name = save_path + file_name_pattern + str(agent_id) + ".csv"
    print("Saving data to {0}".format(created_file_name))
    agent_data.to_csv(created_file_name)

    # Step 3: Simulate agents with recovered parameters
    for sim_agent_id in range(n_agents):
        print('Simulating {0} agent {1} with parameters {2} recovered from participant {3}'.format(
            agent_stuff['learning_style'], sim_agent_id, np.round(rec_pars, 2), agent_id))
        agent_data = fit_params.simulate_agent(all_pars=rec_pars, all_Q_columns=False)

        created_file_name = save_path + file_name_pattern + str(agent_id) + '_' + str(sim_agent_id) + ".csv"
        print("Saving data to {0}".format(created_file_name))
        agent_data.to_csv(created_file_name)


# Play the game to test everything
if interactive_game:
    agent_stuff['id'] = agent_start_id

    # Adjust parameters
    for feature_name in ['learning_style', 'mix_probs']:
        feat = input('{0} (leave blank for "{1}"):'.format(feature_name, agent_stuff[feature_name]))
        if feat:
            agent_stuff[feature_name] = feat
    for par_name in parameters['par_names']:
        par = input('{0} (leave blank for {1}):'.format(par_name, gen_pars[np.array(parameters['par_names']) == par_name]))
        if par:
            gen_pars[np.array(parameters['par_names']) == par_name] = par

    print('\tAGENT CHARACTERISTICS:\n'
          'Learning style: {0}\nMix probs: {1}\nParameters: {2}'.format(
          str(agent_stuff['learning_style']), str(agent_stuff['mix_probs']), str(np.round(gen_pars, 2))))
    fit_params = FitParameters(parameters=parameters,
                               task_stuff=task_stuff,
                               comp_stuff=comp_stuff,
                               agent_stuff=agent_stuff)
    agent_data = fit_params.get_agent_data(way='interactive',
                                           all_pars=gen_pars)
