import numpy as np
from calculate_NLL import ModelFitting
from scipy.optimize import minimize

agents = np.array([15, 16, 17, 18, 20, 22, 23, 24, 25, 31, 32, 33, 34, 35, 40, 42, 43, 44, 48, 49, 52, 53, 56, 60, 63, 65, 67])

agent_stuff = {'learning_style': 'Bayes',
               'method': 'direct',
               'free_par': [True, True, False, False, False],
               'default_par': [0, 1, 0, 0, 0],
               'n_agents': 30,
               'data_path': 'C:/Users/maria/MEGAsync/SLCNdata/PSResults'}

task_stuff = {'n_actions': 2,
              'p_reward': 0.75,
              'path': 'C:/Users/maria/MEGAsync/SLCN/ProbabilisticSwitching/Prerandomized sequences'}

model = ModelFitting(agent_stuff, task_stuff)
params0 = 0.5 * np.ones(len(agent_stuff['free_par']))
result = minimize(model.calculate_fit,
                  params0, (agents[0], True),  # arguments to model.calculate_fit: id, only_NLL
                  method='Nelder-Mead', options={'disp': True})
fit_par = agent_stuff['default_par']
j = 0
for i, par in enumerate(fit_par):
    if agent_stuff['free_par'][i]:
        fit_par[i] = result.x[j]
        j += 1
NLL, BIC, AIC = model.calculate_fit(fit_par, agents[0], False)
model.simulate_agents(fit_par)

# agent_stuff = {'data_path': 'C:/Users/maria/MEGAsync/SLCNdata/PSResults',
#                'alpha': 0.7,
#                'beta': 1,  # 1 for neutral
#                'epsilon': 0,  # 0 for neutral
#                'perseverance': 0,  # 0 for neutral
#                'decay': 0,  # 0 for neutral
#                'method': 'direct'}  # 'direct' for neutral
#
# for ag in agents:
#     print('agent', ag)
#     agent_stuff['id'] = ag
#     task = Task(task_stuff, agent_stuff, goal, ag, n_trials)
#     agent = UniversalAgent(agent_stuff, task, goal, learning_style)
#     hist = History(task, agent)
#
#     for trial in range(1, task.n_trials):
#         task.switch_box(trial, goal)
#         action = agent.take_action(trial, goal)
#         reward = task.produce_reward(action, trial, goal)
#         agent.learn(action, reward)
#         hist.update(agent, task, action, reward, trial)
#
#     hist.save_csv()
