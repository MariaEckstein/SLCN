def run(file_name=None, run_on_cluster=False, main_part=True, plot_heatmaps=False, interactive_game=False):
    import numpy as np
    import glob
    import os
    import pandas as pd
    from fit_parameters import FitParameters
    from minimizer_heatmap import PlotMinimizerHeatmap

    # Things to keep in mind / knobs to turn;
    # separate alphas / betas for different phases
    # suppress at level of policy (i.e., within softmax Q_to_p)

    # TDs / bugs
    # Calculate Q(TS) by hand from Q_low for flat agents in competition phase (alien-same-season)!
    # Record the right Q_high in cloudy season in alien_record_data (index 4)!
    # Save Q and p for competition phase?
    # Fix context and self.context!

    # Model parameters
    fit_par_names = ['alpha', 'beta', 'beta_high']
    learning_style = 'hierarchical'
    mix_probs = True

    # Adjust exact modeling
    if main_part:
        fit_model = True
        simulate_agents = True
        use_existing_data = False
        use_humans = False
        set_specific_parameters = False
        n_agents = 100
        agent_start_id = 700

    # Don't touch
    if run_on_cluster:
        base_path = '/home/bunge/maria/Desktop/Aliens'
        human_data_path = base_path + '/humanData/'
    else:
        base_path = 'C:/Users/maria/MEGAsync/Berkeley/TaskSets'
        human_data_path = base_path + '/Data/version3.1/'
    agent_data_path = base_path + '/AlienGenRec/'
    plot_save_path = agent_data_path + '/heatmaps/'

    # How should the function be minimized?
    minimizer_stuff = {'save_plot_data': True,
                       'create_plot': not run_on_cluster,
                       'plot_save_path': plot_save_path,
                       'verbose': True,
                       'brute_Ns': 4,
                       'hoppin_T': 10.0,
                       'hoppin_stepsize': 0.5,
                       'NM_niter': 3,
                       'NM_xatol': .1,
                       'NM_fatol': .1,
                       'NM_maxfev': 10}

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
                   'beta_scaler': 2,
                   'beta_high_scaler': 4,
                   'learning_style': learning_style,
                   'mix_probs': mix_probs}

    parameters = {'fit_par_names': fit_par_names,
                  'par_names':
                  ['alpha', 'alpha_high', 'beta', 'beta_high', 'epsilon', 'forget', 'create_TS_biased_prefer_new', 'create_TS_biased_copy_old'],
                  'par_hard_limits':  # no values fitted outside; beta will be multiplied by 6 inside of alien_agents.py!
                  ((0., 1.), (0., 1.), (0., 1.), (0., 1.), (0., 1.),  (0., 1.), (0., 1.), (0., 1.)),
                  'par_soft_limits':  # no simulations outside
                  ((0., .5), (0., .5), (0., 1.), (0., 1.), (0., .25), (0., .1), (0., 1.), (0., 1.)),
                  'default_pars':  # when a parameter is fixed
                  np.array([.1, 0.,     .1,       0.,       0.,        0.,       0.,       0.])}

    # Adjust things to user selection
    if main_part:
        if use_existing_data:
            if use_humans:
                save_path = human_data_path
                file_name_pattern = 'aliens'
            else:
                save_path = agent_data_path
                file_name_pattern = 'sim_'
        else:
            save_path = agent_data_path
            file_name_pattern = 'sim_'

        parameters['fit_pars'] = np.array([par in parameters['fit_par_names'] for par in parameters['par_names']])
        fit_par_col_name = '_'.join([agent_stuff['learning_style'], '_'.join(parameters['fit_par_names'])])
        agent_stuff['fit_par'] = fit_par_col_name

        # Create folder to save output files
        if not os.path.isdir(save_path + '/fit_par/'):
            os.makedirs(save_path + '/fit_par/')

        # Create / fit each agent / person
        # Get agent id
        if use_existing_data:
            print(file_name)
            agent_id = int(file_name.split(file_name_pattern)[1].split('.csv')[0])
        else:
            agent_id = file_name
        print('\tPARTICIPANT {0}'.format(agent_id))
        agent_stuff['id'] = agent_id

        # Specify model
        fit_params = FitParameters(parameters=parameters, task_stuff=task_stuff,
                                   comp_stuff=comp_stuff, agent_stuff=agent_stuff)

        # STEP 1: Get data
        if use_existing_data:
            print("Loading data {0}".format(file_name))
            agent_data = pd.read_csv(file_name)

        elif simulate_agents:
            gen_pars = np.array([lim[0] + np.random.rand() * (lim[1] - lim[0]) for lim in parameters['par_soft_limits']])
            gen_pars[np.invert(parameters['fit_pars'])] = 0
            if set_specific_parameters:
                for i, par in enumerate(gen_pars):
                    change_par = input('Accept {0} of {1}? If not, type a new number.'.
                                       format(parameters['par_names'][i], np.round(par, 2)))
                    if change_par:
                        gen_pars[i] = float(change_par)
            print('Simulating {0} agent {1} with parameters {2}'.format(
                agent_stuff['learning_style'], agent_id, np.round(gen_pars, 3)))
            agent_data = fit_params.simulate_agent(all_pars=gen_pars)

            created_file_name = save_path + file_name_pattern + str(agent_id) + ".csv"
            print("Saving simulated data to {0}".format(created_file_name))
            agent_data.to_csv(created_file_name)
        agent_data['sID'] = agent_id  # redundant but crashes otherwise...

        # STEP 2: Fit parameters
        if fit_model:

            # Clean data
            if use_humans:
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

            if 'alpha' in agent_data.columns:
                # Look up generated parameters and display
                gen_pars = agent_data.loc[0, parameters['par_names']]
                gen_pars[2:4] /= agent_stuff['beta_scaler']  # beta
                print("True parameters: {0}".format(np.round(gen_pars.values.astype(np.double), 3)))

            # Find parameters that minimize NLL
            rec_pars = fit_params.get_optimal_pars(agent_data=agent_data, minimizer_stuff=minimizer_stuff)
            print("Fitted parameters: {0}".format(np.round(rec_pars, 3)))

            # Calculate NLL,... of minimizing parameters
            agent_data = fit_params.calculate_NLL(vary_pars=rec_pars[np.argwhere(parameters['fit_pars'])],
                                                  agent_data=agent_data,
                                                  goal='add_decisions_and_fit',
                                                  suff='_rec')

        # Write agent_data as csv
        created_file_name = save_path + '/fit_par/' + file_name_pattern + str(agent_id) + ".csv"
        print("Saving fitted data to {0}".format(created_file_name))
        agent_data.to_csv(created_file_name)

        # STEP 3: Simulate agents with recovered parameters
        if use_humans and simulate_agents:
            for sim_agent_id in range(n_agents):
                print('Simulating {0} agent {1} with parameters recovered from participant {3} {2}'.format(
                    agent_stuff['learning_style'], sim_agent_id, np.round(rec_pars, 3), agent_id))
                agent_data = fit_params.simulate_agent(all_pars=rec_pars)

                created_file_name = save_path + '/fit_par/' + file_name_pattern + str(agent_id) + '_' + str(sim_agent_id) + ".csv"
                print("Saving data to {0}".format(created_file_name))
                agent_data.to_csv(created_file_name)

    # Plot heatmaps from saved data
    if plot_heatmaps:

        # Create folder to save plots
        if not os.path.isdir(plot_save_path):
            os.makedirs(plot_save_path)

        for id in range(400, 450):
            file_name = minimizer_stuff['plot_save_path'] + '/ID' + str(id)
            plot_heatmap = PlotMinimizerHeatmap(file_name)
            plot_heatmap.plot_3d(parameters['fit_par_names'])

    # Play the game interactively to test agents
    if interactive_game:
        agent_stuff['id'] = agent_start_id

        # Adjust parameters
        for feature_name in ['learning_style', 'mix_probs']:
            feat = input('{0} (Hit enter for "{1}"):'.format(feature_name, agent_stuff[feature_name]))
            if feat:
                agent_stuff[feature_name] = feat
        for par_name in parameters['par_names']:
            par = input('{0} (Hit enter for {1}):'.format(par_name, gen_pars[np.array(parameters['par_names']) == par_name]))
            if par:
                gen_pars[np.array(parameters['par_names']) == par_name] = par

        print('\tAGENT CHARACTERISTICS:\n'
              'Learning style: {0}\nMix probs: {1}\nParameters: {2}'.format(
              str(agent_stuff['learning_style']), str(agent_stuff['mix_probs']), str(np.round(gen_pars, 2))))
        fit_params = FitParameters(parameters=parameters,
                                   task_stuff=task_stuff,
                                   comp_stuff=comp_stuff,
                                   agent_stuff=agent_stuff)
        agent_data = fit_params.simulate_agent(all_pars=gen_pars, interactive=True)
