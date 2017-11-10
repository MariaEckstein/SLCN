import numpy as np
from universal_agent import UniversalAgent
from task import Task
from history import History


class ModelFitting(object):

    def __init__(self, agent_stuff, task_stuff):
        self.agent_stuff = agent_stuff
        self.task_stuff = task_stuff

    def simulate_agents(self, ag, params, path):

        goal = 'simulate'

        task = Task(self.task_stuff, params, goal, ag, 200)
        agent = UniversalAgent(self.agent_stuff, params, task, ag)
        hist = History(task, agent, path)

        for trial in range(1, task.n_trials):
            task.switch_box(trial, goal)
            action = agent.take_action(trial, goal)
            reward = task.produce_reward(action, trial, goal)
            agent.learn(action, reward)
            hist.update(agent, task, action, reward, trial)

        hist.save_csv()

    def calculate_fit(self, params, ag, only_NLL):

        goal = 'model'

        task = Task(self.task_stuff, self.agent_stuff, goal, ag, 200)
        agent = UniversalAgent(self.agent_stuff, params, task, ag)

        for trial in range(1, task.n_trials):
            task.switch_box(trial, goal)
            action = agent.take_action(trial, goal)
            reward = task.produce_reward(action, trial, goal)
            agent.learn(action, reward)

        n_fit_params = sum(self.agent_stuff['free_par'])

        BIC = - 2 * agent.LL + n_fit_params * np.log(task.n_trials)
        AIC = - 2 * agent.LL + n_fit_params

        if only_NLL:
            return -agent.LL
        else:
            return [-agent.LL, BIC, AIC]

    def get_fit_par(self, minimization):
        fit_par = self.agent_stuff['default_par']
        j = 0
        for i, par in enumerate(fit_par):
            if self.agent_stuff['free_par'][i]:
                fit_par[i] = minimization.x[j]
                j += 1
        return fit_par
