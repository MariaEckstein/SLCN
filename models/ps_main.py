from universal_agent import UniversalAgent
from task import Task
from history import History
import numpy as np

goal = 'produce_data'  # can be 'model_data' or 'produce_data'
model_type = 'RL'  # can be 'RL' or 'Bayes'
n_trials = 150
if goal == 'model_data':
    agents = np.array([15, 16, 17, 18, 20, 22, 23, 24, 25, 31, 32, 33, 34, 35, 40, 42, 43, 44, 48, 49, 52, 53, 56,
                       60, 63, 65, 67])
else:
    agents = range(30)

task_stuff = {'n_actions': 2,
              'p_reward': 0.75,
              'path': 'C:/Users/maria/MEGAsync/SLCN/ProbabilisticSwitching/Prerandomized sequences'}
agent_stuff = {'data_path': 'C:/Users/maria/MEGAsync/SLCNdata/PSResults',
               'alpha': 0.7,
               'beta': 1,  # 1 for neutral
               'epsilon': 0,  # 0 for neutral
               'perseverance': 0,  # 0 for neutral
               'decay': 0,  # 0 for neutral
               'method': 'direct'}  # 'direct' for neutral

for ag in agents:
    print('agent', ag)
    agent_stuff['id'] = ag
    task = Task(task_stuff, agent_stuff, goal, ag, n_trials)
    agent = UniversalAgent(agent_stuff, task, goal, model_type)
    hist = History(task, agent)

    for trial in range(1, task.n_trials):
        task.switch_box(trial)
        action = agent.take_action(trial)
        reward = task.produce_reward(action, trial)
        agent.learn(action, reward)
        hist.update(agent, task, action, reward, trial)

    hist.save_csv()

# from calculate_NLL import ModelFitting
# from scipy.optimize import minimize
# import numpy as np
# model = ModelFitting()
# bounds = ((0, 1), (0, 30), (0, 1), (0, 30), (0, 1))
# n_params = 5
# params0 = 0.5 * np.ones(n_params)
# model.calculate_NLL(params0, 15, 'RL')
# # algos that can handle bounds: L-BFGS-B; TNC; SLSQP -> all give shitty results
# res = minimize(model.calculate_NLL, params0, (15, 'RL'), method='Nelder-Mead', options={'disp': True})  # , bounds=bounds
