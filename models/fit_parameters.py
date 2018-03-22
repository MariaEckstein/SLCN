import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import brute
from alien_task import Task
from alien_agents import Agent
from alien_record_data import RecordData


class FitParameters(object):
    def __init__(self, parameters, task_stuff, agent_stuff):
        self.parameters = parameters
        self.task_stuff = task_stuff
        self.agent_stuff = agent_stuff
        self.n_fit_par = sum(parameters.fit_pars)

    def get_agent_data(self, way, data_path=(), all_params_lim=()):
        if way == 'simulate':
            return self.simulate_agent(all_params_lim)
        else:
            file_name = data_path + '/PS_' + str(self.agent_stuff['id']) + '.csv'
            return pd.read_csv(file_name)

    def simulate_agent(self, all_params_lim):

        task = Task(self.task_stuff, self.agent_stuff['id'])
        agent = Agent(self.agent_stuff, all_params_lim, self.task_stuff)
        record_data = RecordData(n_trials=task.n_trials,
                                 agent_id=agent.id,
                                 mode='create_from_scratch')

        for trial in range(task.n_trials):
            task.prepare_trial(trial)
            stimulus = task.present_stimulus(trial)
            action = agent.select_action(stimulus)
            reward = task.produce_reward(action)
            agent.learn(stimulus, action, reward)
            record_data.add_behavior(task, stimulus, action, reward, trial)
            record_data.add_decisions(agent, trial)

        record_data.add_parameters(agent)
        return record_data.get()

    def calculate_NLL(self, params_inf, agent_data, goal='calculate_NLL'):

        all_params_lim = self.parameters.inf_to_lim(params_inf)
        agent = Agent(self.agent_stuff, all_params_lim, self.task_stuff)
        if goal == 'add_decisions_and_fit':
            record_data = RecordData(n_trials=np.nan,
                                     agent_id=agent.id,
                                     mode='add_to_existing_data',
                                     agent_data=agent_data)

        n_trials = len(agent_data)
        # for trial in range(n_trials):
        #     action = int(agent_data['selected_box'][trial])
        #     reward = int(agent_data['reward'][trial])
        #     agent.learn(action, reward)
        #     if goal == 'add_decisions_and_fit':
        #         record_data.add_decisions(agent, trial, suff='_rec')

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
        # x0 = brute(func=self.calculate_NLL,
        #            ranges=self.parameters.par_hard_limits,  # should be in the form (slice(-4, 4, 0.25), slice(-4, 4, 0.25))
        #            args=agent_data,
        #            full_output=True,
        #            finish=None)  # should try! https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brute.html#scipy.optimize.brute
        # optimized_params = self.parameters.inf_to_lim(x0)
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
        optimized_params = self.parameters.inf_to_lim(pars_inf[1:])
        return optimized_params

    def write_agent_data(self, agent_data, save_path):
        agent_data.to_csv(save_path + "PS_" + str(agent_data['sID'][0]) + ".csv")
