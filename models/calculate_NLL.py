import numpy as np
from universal_agent import UniversalAgent
from task import Task
from history import History


class ModelFitting(object):

    def __init__(self, task_stuff, agent_stuff):
        self.agent_stuff = agent_stuff
        self.task_stuff = task_stuff

    def simulate_agents(self, params):

        goal = 'simulate'

        for id in range(self.agent_stuff['n_agents']):
            task = Task(self.task_stuff, params, goal, id, 200)
            agent = UniversalAgent(self.agent_stuff, params, task, goal, id)
            hist = History(task, agent)

            for trial in range(1, task.n_trials):
                task.switch_box(trial, goal)
                action = agent.take_action(trial, goal)
                reward = task.produce_reward(action, trial, goal)
                agent.learn(action, reward)
                hist.update(agent, task, action, reward, trial)

            hist.save_csv()

    def calculate_fit(self, params, id, only_NLL):

        goal = 'model'

        task = Task(self.task_stuff, params, goal, id, 200)
        agent = UniversalAgent(self.agent_stuff, params, task, goal, id)

        for trial in range(1, task.n_trials):
            task.switch_box(trial, goal)
            action = agent.take_action(trial, goal)
            reward = task.produce_reward(action, trial, goal)
            agent.learn(action, reward)

        n_fit_params = sum(self.agent_stuff['fit_params'])

        BIC = - 2 * agent.LL + n_fit_params * np.log(task.n_trials)
        AIC = - 2 * agent.LL + n_fit_params

        if only_NLL:
            return -agent.LL
        else:
            return [-agent.LL, BIC, AIC]
