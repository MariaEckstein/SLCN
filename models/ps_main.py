import numpy as np
from parameters import Parameters
from fit_parameters import FitParameters
from gen_rec import GenRec
import glob

# TD: start at the right trial - ie, after explanation!

# Set parameters
n_iter = 15
n_agents = 30
base_path = 'C:/Users/maria/MEGAsync/SLCN'
data_path = base_path + '/data/PSResults'
task_stuff = {'n_actions': 2,
              'p_reward': 0.75,
              'n_trials': 200,
              'av_run_length': 10,  # DOUBLE-CHECK!!!!
              'path': base_path + '/ProbabilisticSwitching/Prerandomized sequences'}
parameters = Parameters(par_names=['alpha', 'beta', 'epsilon', 'perseverance', 'decay'],
                        fit_pars=[True, True, True, True, True],  # which parameters will be fitted?
                        # good limits for simulation: ((0, 1), (1, 6), (0, 0.25), (-0.3, 0.3), (0, 0.3))
                        # limits for fitting ('firsttry'): ((0, 1), (1, 15), (0, 1), (-1, 1), (0, 1))
                        # limits for fitting ('secondtry'): ((0, 1), (1, 10), (0, 0.5), (-0.5, 0.5), (0, 0.5))
                        par_hard_limits=((0, 1), (1, 15), (0, 1), (-1, 1), (0, 1)),
                        par_soft_limits=((0, 1), (1, 6), (0, 0.25), (-0.3, 0.3), (0, 0.3)),
                        default_pars_lim=np.array([0.25, 1+1e-10, 1e-10, 1e-10, 1e-10]))

# Generate and recover
gen_rec = GenRec(parameters=parameters,
                 save_path=base_path + '/models/genrec')
agent_stuff = {}
agent_ids = range(1000, 1000 + n_agents)
for agent_id in agent_ids:
    agent_stuff['id'] = agent_id
    for learning_style in ['Bayes', 'RL']:
        agent_stuff['learning_style'] = learning_style
        for method in ['direct', 'softmax', 'epsilon-greedy']:
            agent_stuff['method'] = method

    parameters.adjust_fit_pars(learning_style=agent_stuff['learning_style'],
                               method=agent_stuff['method'])
    fit_params = FitParameters(parameters=parameters,
                               data_path=data_path,
                               task_stuff=task_stuff,
                               agent_stuff=agent_stuff)
    gen_pars = parameters.create_random_params(scale='lim', get_all=True, mode='soft')  #parameters.default_pars_lim.copy()  #
    agent_data = fit_params.get_agent_data(way='simulate',
                                           all_params_lim=gen_pars)
    rec_pars = fit_params.get_optimal_pars(agent_data=agent_data,
                                           n_iter=n_iter)
    agent_data = fit_params.calculate_NLL(params_inf=parameters.lim_to_inf(rec_pars),
                                          agent_data=agent_data,
                                          goal='add_decisions_and_fit')
    fit_params.write_agent_data(agent_data)
    fit = fit_params.calculate_NLL(params_inf=parameters.lim_to_inf(rec_pars),
                                   agent_data=agent_data,
                                   goal='calculate_fit')
    gen_rec.update_and_save_genrec(gen_pars=gen_pars,
                                   rec_pars=rec_pars,
                                   fit=fit,
                                   agent_stuff=agent_stuff)

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
