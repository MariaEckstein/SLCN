import numpy as np
import pandas as pd
from universal_agent import UniversalAgent
from task import Task
from history import History
from scipy.optimize import minimize
from transform_pars import TransformPars


class ModelFitting(object):

    def __init__(self, agent_stuff, task_stuff, path_extension=''):
        self.agent_stuff = agent_stuff
        self.task_stuff = task_stuff
        self.pars = ['alpha', 'beta', 'epsilon', 'perseverance', 'decay']
        self.genrec = pd.DataFrame(columns=['sID', 'learning_style', 'method', 'NLL', 'BIC', 'AIC'] +
                                           [par + '_gen' for par in self.pars] + [par + '_rec' for par in self.pars])
        self.genrec_row = 0
        self.agent_stuff['data_path'] = 'C:/Users/maria/MEGAsync/SLCNdata/' + self.agent_stuff['learning_style'] +\
                                        '/' + self.agent_stuff['method'] + path_extension

    def adjust_free_par(self):
        if self.agent_stuff['learning_style'] == 'Bayes':
            self.agent_stuff['free_par'][0] = False  # alpha
        else:
            self.agent_stuff['free_par'][0] = True  # alpha

        if self.agent_stuff['method'] == 'epsilon-greedy':
            self.agent_stuff['free_par'][1:3] = [False, True]  # beta, epsilon
        elif self.agent_stuff['method'] == 'softmax':
            self.agent_stuff['free_par'][1:3] = [True, False]  # beta, epsilon
        elif self.agent_stuff['method'] == 'direct':
            self.agent_stuff['free_par'][1:3] = [False, False]  # beta, epsilon

    def update_genrec(self, row, file_path):
        self.genrec.loc[self.genrec_row, :] = row
        self.genrec_row += 1
        self.genrec.to_csv(file_path)

    def simulate_agents(self, params, ag, goal='simulate'):

        task = Task(self.task_stuff, goal, ag, 200, self.agent_stuff)
        agent = UniversalAgent(self.agent_stuff, params, task, ag)
        if goal == 'simulate':
            hist = History(task, agent)

        for trial in range(1, task.n_trials):
            task.switch_box(trial, goal)
            action = agent.take_action(trial, goal)
            reward = task.produce_reward(action, trial, goal)
            agent.learn(action, reward)
            if goal == 'simulate':
                hist.update(agent, task, action, reward, trial)

        n_fit_params = sum(self.agent_stuff['free_par'])
        BIC = - 2 * agent.LL + n_fit_params * np.log(task.n_trials)
        AIC = - 2 * agent.LL + n_fit_params

        if goal == 'calculate_NLL':
            return -agent.LL
        elif goal == 'calculate_fit':
            return [-agent.LL, BIC, AIC]
        elif goal == 'simulate':
            hist.save_csv([-agent.LL, BIC, AIC])

    def minimize_NLL(self, ag, n_fit_par, n_iter):
        trans = TransformPars()
        values = np.zeros([n_iter, n_fit_par + 1])
        for iter in range(n_iter):
            start_par = trans.inverse_sigmoid(np.random.rand(n_fit_par))  # make 0 to 1 into -inf to inf
            minimization = minimize(self.simulate_agents, x0=start_par, args=(ag, 'calculate_NLL'), method='Nelder-Mead')
            values[iter, :] = np.concatenate(([minimization.fun], minimization.x))
        minimum = values[:, 0] == min(values[:, 0])
        best_par = values[minimum][0]
        return best_par[1:]
