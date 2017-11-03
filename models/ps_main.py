import numpy as np

from rl_agent import RLAgent
from bayes_agent import BayesAgent
from task import Task
from history import History

n_trials = 100
n_agents = 30
l_episodes = np.random.choice(range(7, 13), 50)
agent_stuff = {'alpha': 1,
               'beta': 0.2,
               'epsilon': 0.2,
               'prior': 1 / np.mean(l_episodes)}
task_stuff = {'n_actions': 2,
              'reward_prob': 0.75,
              'l_episodes': l_episodes}

for ag in range(n_agents):
    agent_stuff['id'] = ag
    agent = RLAgent(agent_stuff)
    # agent = BayesAgent(agent_stuff, task_stuff)
    task = Task(task_stuff)
    hist = History(task, n_trials)

    for trial in range(n_trials):
        action = agent.take_action()
        reward = task.produce_reward(action)
        switch = task.switch_box()
        agent.learn(action, reward)
        hist.update(agent, task, action, reward, switch, trial)

    hist.transform_into_human_format(agent.id)
    hist.save_csv(agent.id)