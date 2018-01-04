import numpy as np
import pandas as pd
from scipy.optimize import minimize
import os
from task import Task
from universal_agent import UniversalAgent
from record_data import RecordData


class FitParameters(object):
    def __init__(self, parameters, data_path, task_stuff, agent_stuff):
        self.parameters = parameters
        self.data_path = data_path
        if not os.path.isdir(self.data_path):
            os.makedirs(self.data_path)
        self.task_stuff = task_stuff
        self.agent_stuff = agent_stuff
        self.n_fit_par = sum(parameters.fit_pars)

    def get_agent_data(self, way, all_params_lim=()):
        if way == 'simulate':
            return self.simulate_agent(all_params_lim)
        else:
            file_name = self.data_path + '/PS_' + str(self.agent_stuff['id']) + '.csv'
            return pd.read_csv(file_name)

    def simulate_agent(self, all_params_lim):

        task = Task(self.task_stuff, self.agent_stuff['id'])
        agent = UniversalAgent(self.agent_stuff, all_params_lim, self.task_stuff)
        record_data = RecordData(n_trials=self.task_stuff['n_trials'],
                                 agent_id=agent.id,
                                 mode='create_from_scratch')

        for trial in range(task.n_trials):
            task.switch_box()
            agent.calculate_p_actions()
            action = agent.select_action()
            reward = task.produce_reward(action)
            agent.learn(action, reward)
            record_data.add_behavior(action, reward, trial)
            # record_data.add_behavior(task, action, reward, trial)
            record_data.add_decisions(agent, trial)

        record_data.add_parameters(agent)  # for debugging only!
        return record_data.get()

    def calculate_NLL(self, params_inf, agent_data, goal='calculate_NLL'):

        all_params_lim = self.parameters.inf_to_lim(params_inf)
        agent = UniversalAgent(self.agent_stuff, all_params_lim, self.task_stuff)
        record_data = RecordData(n_trials=self.task_stuff['n_trials'],
                                 agent_id=agent.id,
                                 mode='add_to_existing_data',
                                 agent_data=agent_data)

        n_trials = len(agent_data)
        for trial in range(n_trials):
            agent.calculate_p_actions()
            action = int(agent_data['selected_box'][trial])
            reward = int(agent_data['reward'][trial])
            agent.learn(action, reward)
            if goal == 'add_decisions_and_fit':
                record_data.add_behavior(action, reward, trial, suff='_rec')
                # record_data.add_behavior(task, action, reward, trial, suff='_rec')
                record_data.add_decisions(agent, trial, suff='_rec')

        BIC = - 2 * agent.LL + self.n_fit_par * np.log(n_trials)
        AIC = - 2 * agent.LL + self.n_fit_par

        if goal == 'calculate_NLL':
            return -agent.LL
        elif goal == 'calculate_fit':
            return [-agent.LL, BIC, AIC]
        elif goal == 'add_decisions_and_fit':
            record_data.add_parameters(agent, suff='_rec')
            record_data.add_fit(-agent.LL, BIC, AIC, suff='_rec')
            return record_data.get()

    def get_optimal_pars(self, agent_data, n_iter):
        values = np.zeros([n_iter, self.n_fit_par + 1])
        for iter in range(n_iter):
            start_par = self.parameters.create_random_params(scale='inf', get_all=False)
            minimization = minimize(self.calculate_NLL,
                                    x0=start_par,
                                    args=agent_data,
                                    method='Nelder-Mead')
            values[iter, :] = np.concatenate(([minimization.fun], minimization.x))
        minimum = values[:, 0] == min(values[:, 0])
        pars_inf = values[minimum][0]
        return self.parameters.inf_to_lim(pars_inf[1:])

    def write_agent_data(self, agent_data):
        agent_data.to_csv(self.data_path + "/PS_" + str(agent_data['sID'][0]) + ".csv")
