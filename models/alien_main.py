import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from parameters import Parameters
from fit_parameters import FitParameters
from gen_rec import GenRec
from visualize_agent import VisualizeAgent


# TDs / bugs
# First trial is currently not recorded (the data files are missing the alien in the first trial)
# After a context change, the previous TS should be suppressed -> decrease Q_TS

# What should be done?
interactive_game = True
simulate_agents = False
create_sanity_plots = False
check_genrec_values = False
quick_generate_and_recover = True
generate_and_recover = False
fit_human_data = False

# Model fitting parameters
n_iter = 5
n_agents = 200
agent_start_id = 100
base_path = 'C:/Users/maria/MEGAsync/Berkeley/TaskSets'  # CLUSTER: base_path = '/home/bunge/maria/Desktop/Aliens'
data_path = base_path + '/AlienGenRec/'

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
agent_stuff = {'name': 'alien',
               'n_TS': 3,
               'mix_probs': False}

parameters = Parameters(par_names=['alpha', 'beta', 'epsilon'],  # Rewards <= 10 means than beta is 10 times as much!
                        fit_pars=np.ones(6, dtype=bool),  # which parameters will be fitted?
                        par_hard_limits=((0., 1.),  (1., 15.), (0., 1.)),  # no values fitted outside
                        par_soft_limits=((0., 0.5), (1., 6.),  (0., 0.25)),  # no simulations outside
                        default_pars_lim=np.array([0.1, 1., 0.]))  # when a parameter is fixed
viz_agent = VisualizeAgent(parameters, agent_stuff['name'])

# Play the game to test everything
if interactive_game:
    agent_stuff['id'] = agent_start_id
    agent_stuff['learning_style'] = 'hierarchical'
    agent_stuff['mix_probs'] = False
    # parameter_of_interest = 'beta'
    gen_pars = parameters.default_pars_lim
    gen_pars[np.array(parameters.par_names) == 'epsilon'] = 0

    # Un-comment any of the following lines to adjust parameters:
    # agent_stuff['learning_style'] = input('Learning style ("flat" or "hierarchical"):')
    # agent_stuff['mix_probs'] = input('Mix_probs ("True" or "False"):') == "True"
    # gen_pars = [float(input(par_name)) for par_name in parameters.par_names]
    # gen_pars[np.array(parameters.par_names) == parameter_of_interest] = float(input(parameter_of_interest))
    print('\tAGENT CHARACTERISTICS:\n'
          'Learning style: {0}\nMix probs: {1}\nParameters: {2}'.format(
          str(agent_stuff['learning_style']), str(agent_stuff['mix_probs']), str(np.round(gen_pars, 2))))
    fit_params = FitParameters(parameters=parameters,
                               task_stuff=task_stuff,
                               agent_stuff=agent_stuff)
    agent_data = fit_params.get_agent_data(way='interactive',
                                           all_params_lim=gen_pars)

# Simulate specific agents
if simulate_agents:
    import seaborn as sns

    # Specify which agents will be simulated
    agent_stuff['learning_style'] = 'hierarchical'
    agent_id = agent_start_id

    # Simulate a number of n_agents data sets
    while agent_id < agent_start_id + n_agents:
        gen_rec = GenRec(parameters=parameters,
                         full_genrec_path=data_path + '/genrec.csv')
        agent_data = gen_rec.create_simulated_agent(agent_id, task_stuff, agent_stuff,
                                                    save_agent_path=base_path + '/AlienSimulations/')

        # Test: Flat agent selects correct TS in each context
        sns.lmplot(x='TS', y='context',
                   scatter=True, fit_reg=False,
                   data=agent_data)
        plt.show()

        agent_id += 1

# Check that all parameters are working (plot Qs and check if they make sense)
if create_sanity_plots:
    print("Creating sanity plots...")
    agent_id = agent_start_id
    agent_stuff['learning_style'] = 'hierarchical'
    agent_stuff['mix_probs'] = False
    agent_stuff['id'] = agent_id
    fit_params = FitParameters(parameters=parameters,
                               task_stuff=task_stuff,
                               agent_stuff=agent_stuff)
    viz_agent.plot_Qs('alpha', [0.05, 0.4], fit_params, 'Q')
    # viz_agent.plot_Qs('beta', [0.01, 10], fit_params, 'p')
    # viz_agent.plot_Qs('epsilon', [0.01, 0.4], fit_params, 'p')

# Check that genrec recovers Qs and action probs correctly (based on the actual parameter values)
if check_genrec_values:
    for learning_style in ['flat', 'hierarchical']:
        for mix_probs in [False, True]:
            print('Learning style: {0}, Mix_probs: {1}'.format(learning_style, str(mix_probs)))
            viz_agent.plot_generated_recovered_Qs(task_stuff, learning_style, mix_probs)

# Generate and recover
if quick_generate_and_recover:

    # Specify which agent will be tested
    learning_style = 'hierarchical'
    mix_probs = False

    # Specify where data will be saved
    save_agent_path = data_path
    if not os.path.isdir(save_agent_path):
        os.makedirs(save_agent_path)

    gen_rec = GenRec(parameters=parameters,
                     full_genrec_path=data_path + '/genrec.csv')
    agent_id = agent_start_id
    while agent_id < agent_start_id + n_agents:
        print("Agent " + str(agent_id))

        # Specify model
        fit_pars = np.array(parameters.par_names) != 'epsilon'  # np.ones(len(parameters.par_names), dtype=bool)  #
        agent_stuff['learning_style'] = learning_style
        agent_stuff['mix_probs'] = mix_probs
        agent_stuff['id'] = agent_id

        # Decide which parameters will be fit (others will just be known)
        parameters.set_fit_pars(fit_pars)
        parameters.adjust_fit_pars(learning_style)

        fit_par_names = '_'.join([parameters.par_names[int(i)] for i in np.argwhere(parameters.fit_pars)])
        agent_stuff['fit_par'] = fit_par_names
        fit_params = FitParameters(parameters=parameters,
                                   task_stuff=task_stuff,
                                   agent_stuff=agent_stuff)

        # Simulate agent based on random parameters
        gen_pars = parameters.create_random_params(scale='lim', get_all=True, mode='soft')
        gen_pars[np.array(parameters.par_names) == 'epsilon'] = 0
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
                                    file_name='alien')
        fit = fit_params.calculate_NLL(params_inf=parameters.lim_to_inf(rec_pars),
                                       agent_data=agent_data,
                                       default_pars_lim=gen_pars,
                                       goal='calculate_fit')
        gen_rec.update_and_save_genrec(gen_pars=gen_pars,
                                       rec_pars=rec_pars,
                                       fit=fit,
                                       agent_stuff=agent_stuff)
        agent_id += 1

if generate_and_recover:

    # Specify where data will be saved
    save_agent_path = data_path
    if not os.path.isdir(save_agent_path):
        os.makedirs(save_agent_path)

    gen_rec = GenRec(parameters=parameters,
                     full_genrec_path=data_path + '/genrec.csv')
    agent_id = agent_start_id
    while agent_id < agent_start_id + n_agents:
        for learning_style in ['hierarchical', 'flat']:
            for mix_probs in [True, False]:
                for fit_par in parameters.par_names:
                    fit_pars = np.array(parameters.par_names) == fit_par  # Set one parameter to T, all others F

                    # Specify model
                    agent_stuff['learning_style'] = learning_style
                    agent_stuff['mix_probs'] = mix_probs
                    agent_stuff['id'] = agent_id

                    # Decide which parameters will be fit (others will just be known)
                    parameters.set_fit_pars(fit_pars)
                    parameters.adjust_fit_pars(learning_style)
                    if np.sum(parameters.fit_pars) > 0:  # don't do the following if there are no parameters to fit

                        fit_par_names = '_'.join([parameters.par_names[int(i)] for i in np.argwhere(parameters.fit_pars)])
                        agent_stuff['fit_par'] = fit_par_names
                        fit_params = FitParameters(parameters=parameters,
                                                   task_stuff=task_stuff,
                                                   agent_stuff=agent_stuff)
                        print('\nFree params:', fit_par_names, '- Agent:', agent_id, learning_style, mix_probs)

                        # Simulate agent based on random parameters
                        gen_pars = parameters.create_random_params(scale='lim', get_all=True, mode='soft')
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
                                                    file_name='alien')
                        fit = fit_params.calculate_NLL(params_inf=parameters.lim_to_inf(rec_pars),
                                                       agent_data=agent_data,
                                                       default_pars_lim=gen_pars,
                                                       goal='calculate_fit')
                        gen_rec.update_and_save_genrec(gen_pars=gen_pars,
                                                       rec_pars=rec_pars,
                                                       fit=fit,
                                                       agent_stuff=agent_stuff)
                        agent_id += 1

# Fit parameters to human data
if fit_human_data:
    fit_pars = np.array([0, 0, 0, 0, 0, 0], dtype=bool)
    fit_par_names = '_'.join([parameters.par_names[int(i)] for i in np.argwhere(parameters.fit_pars)])
    agent_stuff['fit_par'] = fit_par_names

    file_names = glob.glob(data_path + '/*Aliens_*.csv')
    agent_ids = np.array([int(file_names[i][46:-4]) for i, _ in enumerate(file_names)])
    for agent_id in agent_ids:
        agent_stuff['id'] = agent_id
        for learning_style in ['hierarchical', 'flat']:
            agent_stuff['learning_style'] = learning_style

    # Specify model
    parameters.adjust_fit_pars(learning_style=agent_stuff['learning_style'])
    fit_params = FitParameters(parameters=parameters,
                               task_stuff=task_stuff,
                               agent_stuff=agent_stuff)

    # Specify where data will be saved
    save_agent_path = data_path + '/humans' + '/' + agent_stuff['learning_style'] + '/' + fit_par_names + '/'
    if not os.path.isdir(save_agent_path):
        os.makedirs(save_agent_path)

    # Get participant data, fit parameters and calculate fit, save
    agent_data = fit_params.get_agent_data(way='real_data')
    rec_pars = fit_params.get_optimal_pars(agent_data=agent_data,
                                           n_iter=n_iter)
    agent_data = fit_params.calculate_NLL(params_inf=parameters.lim_to_inf(rec_pars),
                                          agent_data=agent_data,
                                          goal='add_decisions_and_fit')
    fit_params.write_agent_data(agent_data=agent_data,
                                save_path=save_agent_path)
