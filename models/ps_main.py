import numpy as np
from model_fitting import ModelFitting
from transform_pars import TransformPars


# Set parameters
n_agents = 50
n_iter = 15

# Info about task and agent
task_stuff = {'n_actions': 2,
              'p_reward': 0.75,
              'path': 'C:/Users/maria/MEGAsync/SLCN/ProbabilisticSwitching/Prerandomized sequences'}
pars = ['alpha', 'beta', 'epsilon', 'perseverance', 'decay']
n_par = 5
trans = TransformPars()

# Check that parameters are working
agent_stuff = {'default_par': [-1, -3, -100, -100, -100],  # after transform: [0.27, 0.47, 0, 0, 0]
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
        model.simulate_agents([par_gen], ag)

# Generate and recover
def generate_and_recover(param_names, learning_style, method):
    agent_stuff['free_par'] = [np.any(par == np.array(param_names)) for par in pars]
    agent_stuff['learning_style'] = learning_style
    agent_stuff['method'] = method
    model = ModelFitting(agent_stuff, task_stuff, '_fit_' + ''.join(param_names))
    model.adjust_free_par()
    print(model.agent_stuff['free_par'])
    for ag in range(n_agents):
        n_fit_par = sum(agent_stuff['free_par'])
        print('Par:', param_names, '  Learning:', learning_style, '  Method:', method, '  Agent:', ag,
              '  n_fit_par:', n_fit_par)
        # Generate
        gen_par = trans.inverse_sigmoid(np.random.rand(n_fit_par))  # make 0 to 1 into -inf to inf
        model.simulate_agents(gen_par, ag)
        # Recover
        rec_par = model.minimize_NLL(ag, n_iter)
        # Print and save
        model.update_genrec(gen_par, rec_par, ag)

for learning_style in ['RL', 'Bayes']:
    for method in ['direct', 'softmax', 'epsilon-greedy']:
        generate_and_recover(pars, learning_style, method)

# for learning_style in ['RL', 'Bayes']:
#     for method in ['direct', 'softmax', 'epsilon-greedy']:
#         for param_name in pars:
#             print(param_name)
#             # check_parameters(param_name, agent_stuff, task_stuff, learning_style, method)
#             generate_and_recover([param_name], learning_style, method)


# Fit parameters to actual data
# agents = np.array([15, 16, 17, 18, 20, 22, 23, 24, 25, 31, 32, 33, 34, 35, 40, 42, 43, 44, 48, 49, 52, 53, 56, 60, 63, 65, 67])
#
# params0 = 0.5 * np.ones(len(agent_stuff['free_par']))
# NLL, BIC, AIC = model.calculate_fit(params0, agents[0], False)
# minimization = minimize(model.calculate_fit,
#                         params0, (agents[0], True),  # arguments to model.calculate_fit: id, only_NLL
#                         method='Nelder-Mead', options={'disp': True})
# fit_par = model.get_fit_par(minimization)
# NLL, BIC, AIC = model.calculate_fit(fit_par, agents[0], False)
# model.simulate_agents(fit_par)
