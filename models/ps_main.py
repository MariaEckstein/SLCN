from universal_agent import UniversalAgent
from task import Task
from history import History
import numpy as np

goal = 'produce_data'  # can be 'model_data' or 'produce_data'
n_trials = 150
n_agents = 30
data_files = np.array([15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 31, 32, 33, 34, 35, 40, 42, 43, 44, 48, 49, 52, 53, 56,
                       58, 60, 63, 65, 67])
model_type = 'RL'  # can be 'RL' or 'Bayes'

task_stuff = {'n_actions': 2,
              'p_reward': 0.75,
              'path': 'C:/Users/maria/MEGAsync/SLCN/ProbabilisticSwitching/Prerandomized sequences'}
agent_stuff = {'data_path': 'C:/Users/maria/MEGAsync/SLCNdata/PSResults',
               'alpha': 0.5,
               'beta': 1,  # should be 1-20 or so
               'epsilon': 0.2,
               'perseverance': 1,
               'decay': 0.2,
               'method': 'softmax'}

for ag in range(n_agents):
    task = Task(task_stuff, agent_stuff, goal, ag, n_trials)
    agent_stuff['id'] = ag
    if model_type == 'RL':
        agent_stuff['name'] = 'RL'
        agent = UniversalAgent(agent_stuff, task, goal)
    else:
        agent_stuff['name'] = 'Bayes'
        agent = UniversalAgent(agent_stuff, task, goal)
    hist = History(task, agent)

    for trial in range(task.n_trials):
        task.switch_box(trial)
        action = agent.take_action(trial)
        reward = task.produce_reward(action, trial)
        agent.learn(action, reward)
        hist.update(agent, task, action, reward, trial)

    hist.save_csv(agent.id)
