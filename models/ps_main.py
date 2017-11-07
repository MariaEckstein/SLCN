import numpy as np

from rl_agent import RLAgent
from bayes_agent import BayesAgent
from task import Task
from history import History

n_trials = 100
n_agents = 30
l_episodes = np.random.choice(range(7, 16), 50)
model_type = 'Bayes'
rl_agent_stuff = {'name': 'RL',
                  'alpha': 0.5,
                  'beta': 1,  # should be 1-20 or so
                  'epsilon': 0.2,
                  'perseverance': 1,
                  'method': 'softmax'}
bayes_agent_stuff = {'name': 'Bayes',
                     'initial_switch_prob': 1 / np.mean(l_episodes),
                     'impatience': 0.2}
task_stuff = {'n_actions': 2,
              'reward_prob': 0.75,
              'specifications': 'C:/Users/maria/MEGAsync/SLCN/ProbabilisticSwitching/Prerandomized sequences'}

for ag in range(n_agents):
    if model_type == 'RL':
        rl_agent_stuff['id'] = ag
        agent = RLAgent(rl_agent_stuff, task_stuff)
    else:
        bayes_agent_stuff['id'] = ag
        agent = BayesAgent(bayes_agent_stuff, task_stuff)
    task = Task(task_stuff)
    hist = History(task, n_trials, agent.name)

    for trial in range(n_trials):
        action = agent.take_action()
        reward = task.produce_reward(action)
        switch = task.switch_box()
        agent.learn(action, reward)
        hist.update(agent, task, action, reward, switch, trial)

    hist.transform_into_human_format(agent.id)
    hist.save_csv(agent.id)
