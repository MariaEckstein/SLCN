import numpy as np
from parameters import Parameters
from gen_rec import GenRec

# TD: first trial is missing in the fitted data!

# Set parameters
n_iter = 15
n_agents = 30
base_path = 'C:/Users/maria/MEGAsync/SLCN'
task_stuff = {'n_actions': 2,
              'p_reward': 0.75,
              'path': base_path + '/ProbabilisticSwitching/Prerandomized sequences'}
parameters = Parameters(par_names=['alpha', 'beta', 'epsilon', 'perseverance', 'decay'],
                        # good limits for simulation: ((0, 1), (1, 6), (0, 0.25), (-0.3, 0.3), (0, 0.3))
                        # limits for fitting ('firsttry'): ((0, 1), (1, 15), (0, 1), (-1, 1), (0, 1))
                        # limits for fitting ('secondtry'): ((0, 1), (1, 10), (0, 0.5), (-0.5, 0.5), (0, 0.5))
                        par_limits=((0, 1), (1, 15), (0, 1), (-1, 1), (0, 1)),  # parameter scale
                        default_pars=np.array([0.25, 1+1e-10, 1e-10, 1e-10, 1e-10]))  # parameter scale

# Fit parameters to human data
genrec = GenRec(parameters, base_path)
agent_stuff = {}
agent_stuff['data_path'] = base_path + 'data/PSResults'
import glob
file_names = glob.glob(agent_stuff['data_path'] + '/*PS_*.csv')
agents = np.array([int(file_names[i][46:-4]) for i, bla in enumerate(file_names)])
for learning_style in ['RL', 'Bayes']:
    for method in ['direct', 'softmax', 'epsilon-greedy']:
        agent_stuff['learning_style'] = learning_style
        agent_stuff['method'] = method
        genrec.generate_and_recover(genrec.parameters.par_names, agents, n_iter,
                                    agent_stuff, task_stuff, 'fit_human_data')

# Generate and recover
genrec = GenRec(parameters, base_path)
agent_stuff = {}

for learning_style in ['RL', 'Bayes']:
    for method in ['direct', 'softmax', 'epsilon-greedy']:
        agent_stuff['learning_style'] = learning_style
        agent_stuff['method'] = method
        genrec.generate_and_recover(genrec.parameters.par_names, range(n_agents), n_iter,
                                    agent_stuff, task_stuff)
        for param_name in genrec.parameters.par_names:
            # genrec.vary_parameters(param_name, agent_stuff, task_stuff)
            genrec.generate_and_recover([param_name], range(n_agents), n_iter,
                                        agent_stuff, task_stuff)
