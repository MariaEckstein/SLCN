import numpy as np
import pandas as pd
from universal_agent import UniversalAgent
from task import Task
from history import History
from scipy.optimize import minimize
from transform_pars import TransformPars
trans = TransformPars()


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
        self.agent_stuff['data_path'] = 'C:/Users/maria/MEGAsync/SLCNdata/PSResults'

    def adjust_free_par(self):
        if self.agent_stuff['learning_style'] == 'Bayes':
            self.agent_stuff['free_par'][0] = False  # alpha
        if self.agent_stuff['method'] == 'epsilon-greedy':
            self.agent_stuff['free_par'][1:3] = [False, True]  # beta, epsilon
        elif self.agent_stuff['method'] == 'softmax':
            self.agent_stuff['free_par'][1:3] = [True, False]  # beta, epsilon
        elif self.agent_stuff['method'] == 'direct':
            self.agent_stuff['free_par'][1:3] = [False, False]  # beta, epsilon

    def update_genrec(self, gen_par, rec_par, fit, ag):
        gen_pars = trans.get_pars(self.agent_stuff, gen_par)
        rec_pars = trans.get_pars(self.agent_stuff, rec_par)
        gen_pars_01 = trans.adjust_limits(trans.sigmoid(gen_pars))
        rec_pars_01 = trans.adjust_limits(trans.sigmoid(rec_pars))
        print('gen_pars:', np.round(gen_pars_01, 2), '\nrec_pars:', np.round(rec_pars_01, 2), '\nfit:', fit)
        row = np.concatenate(([ag, self.agent_stuff['learning_style'], self.agent_stuff['method']],
                              fit, gen_pars_01, rec_pars_01))
        self.genrec.loc[self.genrec_row, :] = row
        self.genrec_row += 1
        self.genrec.to_csv(self.agent_stuff['data_path'] + '/genrec.csv')

    def simulate_agent(self, params, ag, goal='simulate'):

        task = Task(self.task_stuff, goal, ag, 200, self.agent_stuff)
        agent = UniversalAgent(self.agent_stuff, params, task, ag)
        if not goal == 'calculate_NLL':
            hist = History(task, agent)

        for trial in range(task.n_trials):
            task.switch_box(trial, goal)
            action = agent.take_action(trial, goal)
            reward = task.produce_reward(action, trial, goal)
            agent.learn(action, reward)
            if not goal == 'calculate_NLL':
                hist.update(agent, task, action, reward, trial)

        n_fit_params = sum(self.agent_stuff['free_par'])
        BIC = - 2 * agent.LL + n_fit_params * np.log(task.n_trials)
        AIC = - 2 * agent.LL + n_fit_params

        if goal == 'calculate_NLL':
            return -agent.LL
        elif goal == 'calculate_fit':
            hist.save_csv([-agent.LL, BIC, AIC])
            return [-agent.LL, BIC, AIC]
        elif goal == 'simulate':
            hist.save_csv([-agent.LL, BIC, AIC])

    def minimize_NLL(self, ag, n_iter):
        n_fit_par = sum(self.agent_stuff['free_par'])
        values = np.zeros([n_iter, n_fit_par + 1])
        for iter in range(n_iter):
            start_par = trans.inverse_sigmoid(np.random.rand(n_fit_par))  # make 0 to 1 into -inf to inf
            minimization = minimize(self.simulate_agent, x0=start_par, args=(ag, 'calculate_NLL'), method='Nelder-Mead')
            values[iter, :] = np.concatenate(([minimization.fun], minimization.x))
        minimum = values[:, 0] == min(values[:, 0])
        rec_par = values[minimum][0]
        return rec_par[1:]
