import numpy as np
from model_fitting import ModelFitting
from transform_pars import TransformPars


# TD: first trial is missing in the fitted data!

# Set parameters
n_agents = 40
n_iter = 15

# Info about task and agent
task_stuff = {'n_actions': 2,
              'p_reward': 0.75,
              'path': 'C:/Users/maria/MEGAsync/SLCN/ProbabilisticSwitching/Prerandomized sequences'}
pars = ['alpha', 'beta', 'epsilon', 'perseverance', 'decay']
n_par = 5
trans = TransformPars()

# Check that parameters are working
agent_stuff = {'default_par': [-1, -100, -100, 0, -100],  # after transform: [0.27, 1, 0, 0, 0]
               'n_agents': 1}  # number of simulated agents
def check_parameters(param_name, agent_stuff, task_stuff, learning_style, method):
    agent_stuff['free_par'] = [par == param_name for par in pars]
    agent_stuff['learning_style'] = learning_style
    agent_stuff['method'] = method
    if param_name == 'perseverance':
        default_epsilon = 0
    else:
        default_epsilon = -100
    agent_stuff['default_par'][2] = default_epsilon
    model = ModelFitting(agent_stuff, task_stuff, '_vary_' + param_name)
    for ag, par in enumerate(np.arange(0.01, 0.99, 0.15)):
        print(par)
        par_gen = trans.inverse_sigmoid(par)
        model.simulate_agent([par_gen], ag)

# Generate and recover
def generate_and_recover(param_names, learning_style, method, agents, use='fit_human_data'):
    agent_stuff['free_par'] = [np.any(par == np.array(param_names)) for par in pars]
    agent_stuff['learning_style'] = learning_style
    agent_stuff['method'] = method
    model = ModelFitting(agent_stuff, task_stuff, '_fit_' + ''.join(param_names))
    model.adjust_free_par()
    for ag in agents:
        n_fit_par = sum(agent_stuff['free_par'])
        print('Fitted parameters:', param_names, '(', n_fit_par, '), ' 'Model:', learning_style, ',', method, ', Agent:', ag)
        # Generate
        if use == 'generate_and_recover':
            gen_par = trans.inverse_sigmoid(np.random.rand(n_fit_par))  # make 0 to 1 into -inf to inf
            model.simulate_agent(gen_par, ag)
        else:
            gen_par = np.full(n_fit_par, np.nan)
        # Recover
        rec_par = model.minimize_NLL(ag, n_iter)
        fit = model.simulate_agent(rec_par, ag, 'calculate_fit')
        # Print and save
        model.update_genrec(gen_par, rec_par, fit, ag)

# for learning_style in ['RL', 'Bayes']:
#     for method in ['direct', 'softmax', 'epsilon-greedy']:
#         generate_and_recover(pars, learning_style, method, range(n_agents), 'generate_and_recover')
#
# for learning_style in ['RL', 'Bayes']:
#     for method in ['direct', 'softmax', 'epsilon-greedy']:
#         for param_name in pars:
#             print(param_name)
#             # check_parameters(param_name, agent_stuff, task_stuff, learning_style, method)
#             generate_and_recover([param_name], learning_style, method, range(n_agents), 'generate_and_recover')


# Fit parameters to actual data
agents = np.array([15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 27, 30, 31, 32, 33, 34, 35, 37, 40, 42, 43, 44, 46, 48,
                   49, 52, 53, 56, 60, 63, 65, 67, 69, 70, 73, 74, 77, 80, 87,
                   300, 301, 302, 304, 305, 306, 308, 309, 310, 311, 314, 316])

for learning_style in ['RL', 'Bayes']:
    for method in ['direct', 'softmax', 'epsilon-greedy']:
        generate_and_recover(pars, learning_style, method, agents)
