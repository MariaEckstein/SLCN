import numpy as np
import pandas as pd
from universal_agent import UniversalAgent
from task import Task
from history import History
from scipy.optimize import minimize


class ModelFitting(object):

    def __init__(self, agent_stuff, task_stuff):
        self.agent_stuff = agent_stuff
        self.task_stuff = task_stuff
        pars = ['alpha', 'beta', 'epsilon', 'perseverance', 'decay']
        self.genrec = pd.DataFrame(columns=['sID', 'learning_style', 'method', 'NLL', 'BIC', 'AIC'] +
                                           [par + '_gen' for par in pars] + [par + '_rec' for par in pars])
        self.genrec_row = 0

    def update_genrec_and_save(self, row, file_path):
        self.genrec.loc[self.genrec_row, :] = row
        self.genrec_row += 1
        self.genrec.to_csv(file_path)

    def simulate_agents(self, ag, params):

        goal = 'simulate'

        task = Task(self.task_stuff, params, goal, ag, 200)
        agent = UniversalAgent(self.agent_stuff, params, task, ag)
        hist = History(task, agent)

        for trial in range(1, task.n_trials):
            task.switch_box(trial, goal)
            action = agent.take_action(trial, goal)
            reward = task.produce_reward(action, trial, goal)
            agent.learn(action, reward)
            hist.update(agent, task, action, reward, trial)

        hist.save_csv()

        return np.array([agent.alpha, agent.beta, agent.epsilon, agent.perseverance, agent.decay])

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

    def get_fit_par(self, params):
        pars = self.agent_stuff['default_par']
        j = 0
        for i, par in enumerate(pars):
            if self.agent_stuff['free_par'][i]:
                pars[i] = params[j]
                j += 1
        return np.array(pars)

    def minimize_NLL(self, ag, n_fit_par, n_iter):
        values = np.zeros([n_iter, n_fit_par + 1])
        for iter in range(n_iter):
            params0 = np.random.rand(n_fit_par)
            minimization = minimize(self.calculate_fit, x0=params0, args=(ag, True), method='Nelder-Mead')
            values[iter, :] = np.concatenate(([minimization.fun], minimization.x))
        minimum = values[:, 0] == min(values[:, 0])
        best_par = values[minimum][0]
        return best_par[1:]
