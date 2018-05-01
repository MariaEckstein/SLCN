import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from parameters import Parameters
from fit_parameters import FitParameters
from gen_rec import GenRec
from visualize_agent import VisualizeAgent


# What should be done?
simulate_agents = True
create_sanity_plots = False
check_genrec_values = False
generate_and_recover = False
fit_human_data = False

# Model fitting parameters
n_iter = 20
n_agents = 30
agent_start_id = 0
base_path = 'C:/Users/maria/MEGAsync/Berkeley/TaskSets'
data_path = base_path + '/AlienGenRec/'

# Task parameters
n_actions = 3
                # TS0
TSs = np.array([[[0, 6, 0],  # alien0, items0-2
                [0, 0, 4],  # alien1, items0-2
                [5, 0, 0],  # etc.
                [10, 0, 0]],
                # TS1
               [[0, 0, 2],  # alien0, items0-2
                [0, 8, 0],  # etc.
                [0, 0, 7],
                [0, 3, 0]],
                # TS2
               [[0, 0, 7],  # TS2
                [3, 0, 0],
                [0, 3, 0],
                [2, 0, 0]]])

task_stuff = {'n_trials_per_alien': 13,  # 13
              'n_blocks': 3,  # 3
              'n_aliens': 4,
              'n_actions': n_actions,
              'n_contexts': 3,
              'TS': TSs}
agent_stuff = {'name': 'alien',
               'n_TS': 3,
               'mix_probs': False}

parameters = Parameters(par_names=['alpha', 'beta', 'epsilon', 'perseverance', 'forget', 'mix'],
                        fit_pars=np.ones(6, dtype=bool),  # which parameters will be fitted?
                        par_hard_limits=((0, 1), (1, 15), (0, 1), (-1, 1), (0, 1), (0, 1)),  # no values fitted outside
                        par_soft_limits=((0, 1), (1, 6), (0, 0.25), (-0.3, 0.3), (0, 0.3), (0, 1)),  # no simul. outside
                        default_pars_lim=np.array([0.05, 1+1e-5, 1e-5, 1e-5, 1e-5, 1]))  # when a parameter is fixed
viz_agent = VisualizeAgent(parameters, agent_stuff['name'])

# Simulate specific agents
if simulate_agents:
    import seaborn as sns

    # Specify which agents will be simulated
    agent_stuff['method'] = 'softmax'
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
    agent_id = agent_start_id
    agent_stuff['method'] = 'epsilon-greedy'
    agent_stuff['learning_style'] = 'flat'
    agent_stuff['id'] = agent_id
    fit_params = FitParameters(parameters=parameters,
                               task_stuff=task_stuff,
                               agent_stuff=agent_stuff)
    viz_agent.plot_Qs('alpha', [0.01, 0.4], fit_params)
    # agent_stuff['method'] = 'softmax'
    # agent_stuff['learning_style'] = 'hierarchical'
    # agent_stuff['id'] = agent_id
    # fit_params = FitParameters(parameters=parameters,
    #                            task_stuff=task_stuff,
    #                            agent_stuff=agent_stuff)
    #
    # viz_agent.plot_Qs('alpha', [0.01, 0.4], fit_params)
    # viz_agent.plot_Qs('beta', [0.1, 1, 10], fit_params)
    # viz_agent.plot_Qs('perseverance', [0.1, 0.99], fit_params)
    # viz_agent.plot_Qs('forget', [0.01, 0.99], fit_params)
    # viz_agent.plot_Qs('mix', [0.01, 0.5, 0.99], fit_params)
    # agent_stuff['method'] = 'epsilon-greedy'
    # viz_agent.plot_Qs('alpha', [0.01, 0.4], fit_params)
    # viz_agent.plot_Qs('epsilon', [0.01, 0.4], fit_params)
    # agent_stuff['method'] = 'softmax'
    # agent_stuff['learning_style'] = 'flat'
    # viz_agent.plot_Qs('alpha', [0.01, 0.4], fit_params)

# Check that genrec recovers Qs and action probs correctly (based on the actual parameter values)
if check_genrec_values:
    for method in ['epsilon-greedy', 'softmax']:
        for learning_style in ['flat', 'hierarchical']:
            viz_agent.plot_generated_recovered_Qs(task_stuff, method, learning_style)

# Generate and recover
if generate_and_recover:

    # Specify where data will be saved
    save_agent_path = data_path
    if not os.path.isdir(save_agent_path):
        os.makedirs(save_agent_path)

    gen_rec = GenRec(parameters=parameters,
                     full_genrec_path=data_path + '/genrec.csv')
    agent_id = agent_start_id
    while agent_id < agent_start_id + n_agents:
        for method in ['softmax', 'epsilon-greedy']:
            for learning_style in ['flat', 'hierarchical']:
                for fit_par in range(len(parameters.par_names)):
                    fit_pars = np.zeros(len(parameters.par_names), dtype=bool)
                    fit_pars[fit_par] = True  # Set fitting for one parameter to True, all others to False

                    # Specify model
                    agent_stuff['method'] = method
                    agent_stuff['learning_style'] = learning_style
                    agent_stuff['id'] = agent_id

                    parameters.set_fit_pars(fit_pars)
                    parameters.adjust_fit_pars(method, learning_style)
                    if np.sum(parameters.fit_pars) > 0:  # don't do the following if there are no parameters to fit
                        fit_par_names = '_'.join([parameters.par_names[int(i)] for i in np.argwhere(parameters.fit_pars)])
                        agent_stuff['fit_par'] = fit_par_names
                        fit_params = FitParameters(parameters=parameters,
                                                   task_stuff=task_stuff,
                                                   agent_stuff=agent_stuff)
                        print('\nFitted parameters:', fit_par_names, '- Agent:', agent_id, method, learning_style)

                        # Create random parameters to simulate data, fit parameters and calculate fit, create genrec
                        gen_pars = parameters.create_random_params(scale='lim', get_all=True, mode='soft')
                        agent_data = fit_params.get_agent_data(way='simulate',
                                                               all_params_lim=gen_pars)
                        rec_pars = fit_params.get_optimal_pars(agent_data=agent_data,
                                                               n_iter=n_iter)
                        agent_data = fit_params.calculate_NLL(params_inf=parameters.lim_to_inf(rec_pars),
                                                              agent_data=agent_data,
                                                              goal='add_decisions_and_fit')
                        fit_params.write_agent_data(agent_data=agent_data,
                                                    save_path=save_agent_path)
                        fit = fit_params.calculate_NLL(params_inf=parameters.lim_to_inf(rec_pars),
                                                       agent_data=agent_data,
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
            for method in ['softmax', 'epsilon-greedy']:
                agent_stuff['method'] = method

    # Specify model
    parameters.adjust_fit_pars(learning_style=agent_stuff['learning_style'],
                               method=agent_stuff['method'])
    fit_params = FitParameters(parameters=parameters,
                               task_stuff=task_stuff,
                               agent_stuff=agent_stuff)

    # Specify where data will be saved
    save_agent_path = data_path + '/humans' + '/' + agent_stuff['learning_style'] + '/' + \
                      agent_stuff['method'] + '/' + fit_par_names + '/'
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
