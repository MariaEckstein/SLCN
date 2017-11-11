import numpy as np
from model_fitting import ModelFitting


task_stuff = {'n_actions': 2,
              'p_reward': 0.75,
              'path': 'C:/Users/maria/MEGAsync/SLCN/ProbabilisticSwitching/Prerandomized sequences'}

# Generate and recover
agent_stuff = {'free_par': np.array([True, False, False, False, False]),  # changeable (T) and fixed (F) parameters
               'default_par': [0, 0.1, 0, 0, 0],  # parameter values if fixed
               'n_agents': 1}  # number of simulated agents

genrec_row = 0
for learning_style in ['RL', 'Bayes']:
    for method in ['softmax', 'direct', 'epsilon-greedy']:
        if method == 'softmax':
            agent_stuff['free_par'][1] = True  # beta
        elif method == 'epsilon-greedy':
            agent_stuff['free_par'][2] = True  # epsilon
        n_fit_par = sum(agent_stuff['free_par'])
        agent_stuff['learning_style'] = learning_style
        agent_stuff['method'] = method
        agent_stuff['data_path'] = 'C:/Users/maria/MEGAsync/SLCNdata/' + agent_stuff['learning_style'] +\
                                   '/' + agent_stuff['method']
        model = ModelFitting(agent_stuff, task_stuff)
        print('Learning:', learning_style, '\nMethod:', method)
        # Generate
        for ag in range(50):
            print('agent', ag)
            rand_par = np.random.rand(n_fit_par)
            sim_par = model.simulate_agents(ag, rand_par, '')
            print('sim_par', sim_par)
        # Recover
            best_par = model.minimize_NLL(ag, n_fit_par, n_iter=50)
            fit_par = model.get_fit_par(best_par)
            print(fit_par)
            NLL, BIC, AIC = model.calculate_fit(fit_par, ag, '', False)
            model.update_genrec_and_save(np.concatenate(([ag, learning_style, method, NLL, BIC, AIC], sim_par, fit_par)),
                                         agent_stuff['data_path'] + '/genrec.csv')

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
