import numpy as np
from parameters import Parameters
from gen_rec import GenRec


# TD: first trial is missing in the fitted data!
# TD: I'm imposing limits when recovering -> without?


# Set parameters
n_iter = 15
n_agents = 30
task_stuff = {'n_actions': 2,
              'p_reward': 0.75,
              'path': 'C:/Users/maria/MEGAsync/SLCN/ProbabilisticSwitching/Prerandomized sequences'}
parameters = Parameters(par_names=['alpha', 'beta', 'epsilon', 'perseverance', 'decay'],
                        # good limits for simulation: ((0, 1), (1, 6), (0, 0.25), (-0.3, 0.3), (0, 0.3))
                        # limits for fitting ('fristtry'): ((0, 1), (1, 15), (0, 1), (-1, 1), (0, 1))
                        # limits for fitting: ((0, 1), (1, 10), (0, 0.5), (-0.5, 0.5), (0, 0.5))
                        par_limits=((0, 1), (1, 10), (0, 0.5), (-0.5, 0.5), (0, 0.5)),  # parameter scale
                        default_pars=np.array([0.25, 1+1e-10, 1e-10, 1e-10, 1e-10]))  # parameter scale

# Fit parameters to human data
# genrec = GenRec(parameters)
# agent_stuff = {}
# agents = np.array([15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 27, 30, 31, 32, 33, 34, 35, 37, 40, 42, 43, 44, 46, 48,
#                    49, 52, 53, 56, 60, 63, 65, 67, 69, 70, 73, 74, 77, 80, 87,
#                    300, 301, 302, 304, 305, 306, 308, 309, 310, 311, 314, 316])
#
# for learning_style in ['RL', 'Bayes']:
#     for method in ['direct', 'softmax', 'epsilon-greedy']:
#         agent_stuff['learning_style'] = learning_style
#         agent_stuff['method'] = method
#         genrec.generate_and_recover(genrec.parameters.par_names, agents, n_iter, agent_stuff, task_stuff)

# Generate and recover
genrec = GenRec(parameters)
agent_stuff = {}

for learning_style in ['RL', 'Bayes']:
    for method in ['direct', 'softmax', 'epsilon-greedy']:
        agent_stuff['learning_style'] = learning_style
        agent_stuff['method'] = method
        genrec.generate_and_recover(genrec.parameters.par_names, range(n_agents), n_iter,
                                    agent_stuff, task_stuff, 'generate_and_recover')
        for param_name in genrec.parameters.par_names:
            # genrec.vary_parameters(param_name, agent_stuff, task_stuff)
            genrec.generate_and_recover([param_name], range(n_agents), n_iter,
                                        agent_stuff, task_stuff, 'generate_and_recover')
