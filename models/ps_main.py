import numpy as np
from parameters import Parameters
from fit_parameters import FitParameters
from gen_rec import GenRec
import glob
import os
from visualize_agent import VisualizeAgent

# TD: start at the right trial - ie, after task instructions!

# What should be done?
simulate_agents = False
create_sanity_plots = False
check_genrec_values = False
generate_and_recover = False
fit_human_data = True

# Set parameters
n_iter = 10
n_agents = 5
agent_start_id = 1000
base_path = 'C:/Users/maria/MEGAsync/SLCN'
data_path = base_path + '/PSGenRec'
raw_data_path = base_path + 'data/PSResults'
task_stuff = {'n_actions': 2,
              'p_reward': 0.75,
              'n_trials': 200,
              'av_run_length': 10,  # DOUBLE-CHECK!!!!
              'path': base_path + '/ProbabilisticSwitching/Prerandomized sequences'}
agent_stuff = {'name': 'PS_'}
parameters = Parameters(par_names=['alpha', 'beta', 'epsilon', 'perseverance', 'decay',
                                   'w_reward', 'w_noreward', 'w_explore'],
                        fit_pars=np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=bool),  # which parameters will be fitted?
                        par_hard_limits=((0, 1), (1, 15), (0, 1), (-1, 1), (0, 1),  # values can't be fitted outside
                                         (-1, 1), (-1, 1), (-1, 1)),
                        par_soft_limits=((0, 1), (1, 6), (0, 0.25), (-0.3, 0.3), (0, 0.3),  # no simulations outside
                                         (-0.1, 0.3), (0.1, 0.5), (-0.5, 0.1)),
                        default_pars_lim=np.array([0.25, 1+1e-10, 1e-10, 1e-10, 1e-10,  # when a parameter is fixed
                                                   0.1, 0.5, -0.2]))
viz_agent = VisualizeAgent(parameters, agent_stuff['name'])
gen_rec = GenRec(parameters=parameters,
                 full_genrec_path=data_path + '/genrec.csv')

# Simulate specific agents
if simulate_agents:

    # Specify which agents will be simulated
    agent_stuff['method'] = 'softmax'
    agent_stuff['learning_style'] = 'Bayes'
    agent_id = agent_start_id

    # Simulate a number of n_agents data sets
    while agent_id < agent_start_id + n_agents:
        agent_data = gen_rec.create_simulated_agent(agent_id, task_stuff, agent_stuff,
                                                    save_agent_path=base_path + '/PSSimulations/')
        agent_id += 1

# Check that all parameters are working (plot Qs and check if they make sense)
if create_sanity_plots:
    agent_id = agent_start_id
    agent_stuff['method'] = 'epsilon-greedy'
    agent_stuff['learning_style'] = 'RL'
    agent_stuff['id'] = agent_id
    fit_params = FitParameters(parameters=parameters,
                               task_stuff=task_stuff,
                               agent_stuff=agent_stuff)
    # viz_agent.plot_Qs('alpha', [0.01, 0.4], fit_params)  # NOT working yet!

# Check that genrec recovers Qs and action probs correctly (based on the actual parameter values)
if check_genrec_values:
    for method in ['epsilon-greedy', 'softmax']:
        for learning_style in ['RL', 'Bayes']:
            viz_agent.plot_generated_recovered_Qs(task_stuff, method, learning_style)

# Generate and recover
if generate_and_recover:

    # Specify where data will be saved
    save_agent_path = data_path
    if not os.path.isdir(save_agent_path):
        os.makedirs(save_agent_path)

    gen_rec = GenRec(parameters=parameters,
                     full_genrec_path=data_path + '/PSgenrec.csv')
    agent_id = agent_start_id
    while agent_id < agent_start_id + n_agents:
        for method in ['softmax', 'epsilon-greedy']:
            for learning_style in ['RL', 'Bayes']:
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
    parameters.fit_pars = np.array([1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

    file_names = glob.glob(raw_data_path + '/*PS_*.csv')
    agent_ids = np.array([int(file_names[i][46:-4]) for i, _ in enumerate(file_names)])
    agent_ids = [agent_id for i, agent_id in enumerate(agent_ids) if agent_id < 1000]
    for agent_id in agent_ids:
        agent_stuff['id'] = agent_id
        for learning_style in ['RL', 'Bayes']:
            agent_stuff['learning_style'] = learning_style
            for method in ['softmax', 'epsilon-greedy']:
                agent_stuff['method'] = method

                # Specify model
                parameters.adjust_fit_pars(learning_style=agent_stuff['learning_style'],
                                           method=agent_stuff['method'])
                fit_par_names = '_'.join([parameters.par_names[int(i)] for i in np.argwhere(parameters.fit_pars)])
                agent_stuff['fit_par'] = fit_par_names
                print('\nFitted parameters:', agent_stuff['fit_par'], '- Agent:', agent_id, method, learning_style)
                fit_params = FitParameters(parameters=parameters,
                                           task_stuff=task_stuff,
                                           agent_stuff=agent_stuff)
                # Get participant data, fit parameters and calculate fit, save
                agent_data = fit_params.get_agent_data(way='real_data',
                                                       data_path=raw_data_path)
                rec_pars = fit_params.get_optimal_pars(agent_data=agent_data,
                                                       n_iter=n_iter)
                print('\nRecovered parameters:', rec_pars)
                agent_data = fit_params.calculate_NLL(params_inf=parameters.lim_to_inf(rec_pars),
                                                      agent_data=agent_data,
                                                      goal='add_decisions_and_fit')
                # Save participant data, included the fitted parameters
                save_agent_path = raw_data_path + '/humans/'
                file_name = agent_stuff['learning_style'] + '_' + \
                            agent_stuff['method'] + '_' + fit_par_names + 'PS'
                if not os.path.isdir(save_agent_path):
                    os.makedirs(save_agent_path)
                fit_params.write_agent_data(agent_data=agent_data,
                                            save_path=save_agent_path,
                                            file_name=file_name)
