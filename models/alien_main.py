import numpy as np
from parameters import Parameters
from fit_parameters import FitParameters
from gen_rec import GenRec
import glob
import os

# TD: start at the right trial - ie, after task instructions!

# Set parameters
context_sequence = range(3)
n_iter = 20
n_agents = 500
agent_start_id = 1000
base_path = 'C:/Users/maria/MEGAsync/SLCN'
data_path = base_path + '/PSGenRec'
task_stuff = {'n_actions': 2,
              'p_reward': 0.75,
              'n_trials': 200,
              'av_run_length': 10,  # DOUBLE-CHECK!!!!
              'path': base_path + '/ProbabilisticSwitching/Prerandomized sequences'}
parameters = Parameters(par_names=['alpha', 'beta', 'epsilon', 'perseverance', 'decay',
                                   'w_reward', 'w_noreward', 'w_explore'],
                        fit_pars=np.array([1, 1, 1, 1, 1, 1, 1], dtype=bool),  # which parameters will be fitted?
                        par_hard_limits=((0, 1), (1, 15), (0, 1), (-1, 1), (0, 1),  # values can't be fitted outside
                                         (-1, 1), (-1, 1), (-1, 1)),
                        par_soft_limits=((0, 1), (1, 6), (0, 0.25), (-0.3, 0.3), (0, 0.3),  # no simulations outside
                                         (-0.1, 0.3), (0.1, 0.5), (-0.5, 0.1)),
                        default_pars_lim=np.array([0.25, 1+1e-10, 1e-10, 1e-10, 1e-10,  # when a parameter is fixed
                                                   0.1, 0.5, -0.2]))
gen_rec = GenRec(parameters=parameters,
                 full_genrec_path=data_path + '/genrec.csv')

# Generate and recover
agent_stuff = {}
agent_id = agent_start_id
while agent_id < agent_start_id + n_agents:
    for learning_style in ['RL', 'Bayes']:  # 'estimate-switch',
        for method in ['softmax', 'epsilon-greedy']:
            for fit_pars in (np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=bool),  # alpha / the three / none + epsilon / b
                             np.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=bool),  # alpha & persev. / persev. only (not RL)
                             np.array([0, 0, 0, 0, 1, 0, 0, 0], dtype=bool),  # alpha & decay / decay only (not RL)
                             np.array([0, 0, 0, 1, 1, 0, 0, 0], dtype=bool)):  # alpha, persev., & decay / persev. & decay (not RL)

                # Specify model to be run (RL or Bayes; softmax or epsilon-greedy; which parameters)
                agent_stuff['learning_style'] = learning_style
                agent_stuff['method'] = method
                agent_stuff['id'] = agent_id

                parameters.set_fit_pars(fit_pars)
                parameters.adjust_fit_pars(learning_style=agent_stuff['learning_style'],
                                           method=agent_stuff['method'])
                fit_par_names = '_'.join([parameters.par_names[int(i)] for i in np.argwhere(parameters.fit_pars)])
                agent_stuff['fit_par'] = fit_par_names
                fit_params = FitParameters(parameters=parameters,
                                           task_stuff=task_stuff,
                                           agent_stuff=agent_stuff)
                print('\nParameters:', fit_par_names, '- Agent', agent_id)

                # Specify where data will be saved
                save_agent_path = data_path + '/' + agent_stuff['learning_style'] + '/' + \
                                  agent_stuff['method'] + '/' + fit_par_names + '/'
                if not os.path.isdir(save_agent_path):
                    os.makedirs(save_agent_path)

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

# # Fit parameters to human data
# agent_stuff = {}
# file_names = glob.glob(data_path + '/*PS_*.csv')
# agent_ids = np.array([int(file_names[i][46:-4]) for i, _ in enumerate(file_names)])
# for agent_id in agent_ids:
#     agent_stuff['id'] = agent_id
#     for learning_style in ['RL', 'Bayes']:
#         agent_stuff['learning_style'] = learning_style
#         for method in ['direct', 'softmax', 'epsilon-greedy']:
#             agent_stuff['method'] = method
#
# parameters.adjust_fit_pars(learning_style=agent_stuff['learning_style'],
#                            method=agent_stuff['method'])
# fit_params = FitParameters(parameters=parameters,
#                            data_path=data_path,
#                            task_stuff=task_stuff,
#                            agent_stuff=agent_stuff)
# agent_data = fit_params.get_agent_data(way='real_data')
# rec_pars = fit_params.get_optimal_pars(agent_data=agent_data,
#                                        n_iter=n_iter)
# agent_data = fit_params.calculate_NLL(params=rec_pars,
#                                       agent_data=agent_data,
#                                       goal='add_decisions_and_fit')
# fit_params.write_agent_data(agent_data)
