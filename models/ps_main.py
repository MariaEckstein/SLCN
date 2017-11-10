import numpy as np
from calculate_NLL import ModelFitting
from scipy.optimize import minimize

task_stuff = {'n_actions': 2,
              'p_reward': 0.75,
              'path': 'C:/Users/maria/MEGAsync/SLCN/ProbabilisticSwitching/Prerandomized sequences'}
n_par = 5

# Generate and recover
agent_stuff = {'free_par': np.full(n_par, True, dtype=bool),  # which parameters are changeable (True) versus fixed?
               'default_par': [0, .1, 0, 0, 0],  # parameter values if fixed
               'n_agents': 1}  # number of simulated agents

for learning_style in ['RL', 'Bayes']:
    for method in ['epsilon-greedy', 'softmax', 'direct']:
        agent_stuff['learning_style'] = learning_style
        agent_stuff['method'] = method
        agent_stuff['data_path'] = 'C:/Users/maria/MEGAsync/SLCNdata/' + agent_stuff['learning_style'] + '/' + agent_stuff['method']
        model = ModelFitting(agent_stuff, task_stuff)
        # Generate
        for ag in range(20):
            model.simulate_agents(ag, agent_stuff['default_par'],'')
            model.simulate_agents(ag, np.random.rand(n_par), '_rand')
        # Recover
            params0 = 0.5 * np.ones(len(agent_stuff['free_par']))
            minimization = minimize(model.calculate_fit,
                                    params0, (ag, True),  # arguments to model.calculate_fit: id, only_NLL
                                    method='Nelder-Mead', options={'disp': True})
            fit_par = model.get_fit_par(minimization)
            NLL, BIC, AIC = model.calculate_fit(fit_par, ag, False)
            model.simulate_agents(ag, fit_par, '_sim')

# Fit parameters to actual data
agents = np.array([15, 16, 17, 18, 20, 22, 23, 24, 25, 31, 32, 33, 34, 35, 40, 42, 43, 44, 48, 49, 52, 53, 56, 60, 63, 65, 67])

params0 = 0.5 * np.ones(len(agent_stuff['free_par']))
NLL, BIC, AIC = model.calculate_fit(params0, agents[0], False)
minimization = minimize(model.calculate_fit,
                        params0, (agents[0], True),  # arguments to model.calculate_fit: id, only_NLL
                        method='Nelder-Mead', options={'disp': True})
fit_par = model.get_fit_par(minimization)
NLL, BIC, AIC = model.calculate_fit(fit_par, agents[0], False)
model.simulate_agents(fit_par)
