import numpy as np
import ast
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import brute
from alien_task import Task
from competition_phase import CompetitionPhase
from alien_agents import Agent
from alien_record_data import RecordData
from simulate_interactive import SimulateInteractive
# from ps_task import Task
# from ps_agent import Agent
# from ps_record_data import RecordData


class FitParameters(object):
    def __init__(self, parameters, task_stuff, comp_stuff, agent_stuff):
        self.parameters = parameters
        self.task_stuff = task_stuff
        self.comp_stuff = comp_stuff
        self.agent_stuff = agent_stuff
        assert self.agent_stuff['name'] in ['alien', 'PS_']
        self.n_fit_par = sum(parameters.fit_pars)

    def get_agent_data(self, way, data_path='', all_Q_columns=False, all_params_lim=()):
        assert(way in ['simulate', 'real_data', 'interactive'])
        if way in ['simulate', 'interactive']:
            return self.simulate_agent(all_params_lim, all_Q_columns, interactive=way == 'interactive')
        elif way == 'real_data':
            return pd.read_csv(data_path)

    def simulate_agent(self, all_params_lim, all_Q_columns, interactive=False):

        task = Task(self.task_stuff)
        agent = Agent(self.agent_stuff, all_params_lim, self.task_stuff)
        record_data = RecordData(agent_id=agent.id,
                                 mode='create_from_scratch',
                                 task=task)
        if interactive:
            sim_int = SimulateInteractive(agent, self.agent_stuff['mix_probs'])

        total_trials = 0
        for phase in ['1InitialLearning', '2CloudySeason', 'Refresher2']:
            print(phase)
            task.set_phase(phase)
            agent.task_phase = phase
            n_trials = int(task.n_trials_per_phase[np.array(task.phases) == task.phase])
            for trial in range(n_trials):
                if interactive:
                    stimulus = sim_int.trial(trial)
                    [task.context, task.alien] = stimulus
                    sim_int.print_values_pre()
                    action = int(input('Action (0, 1, 2):'))
                else:
                    task.prepare_trial(trial)
                    stimulus = task.present_stimulus(trial)
                    action = agent.select_action(stimulus)
                [reward, correct] = task.produce_reward(action)
                agent.learn(stimulus, action, reward)
                if interactive:
                    sim_int.print_values_post(action, reward, correct)
                record_data.add_behavior(task, stimulus, action, reward, correct, total_trials, phase)
                record_data.add_decisions(agent, total_trials, suff='', all_Q_columns=all_Q_columns)
                total_trials += 1

        task.set_phase('3PickAliens')
        print(task.phase)
        comp = CompetitionPhase(self.comp_stuff, self.task_stuff)
        for trial in range(sum(comp.n_trials)):
            comp.prepare_trial(trial)
            stimuli = comp.present_stimulus(trial)
            selected = agent.competition_selection(stimuli, comp.current_phase)
            if interactive:
                print('\tTRIAL {0} ({1}),\nstimuli {2}, values: {3}, probs.: {4}'.format(
                trial, comp.current_phase, stimuli, str(np.round(agent.Q_stimuli, 2)), str(np.round(agent.p_stimuli, 2))))
            record_data.add_behavior_and_decisions_comp(stimuli, selected, agent.Q_stimuli, agent.p_stimuli,
                                                        total_trials, task.phase, comp.current_phase)
            total_trials += 1

        for phase in ['Refresher3', '5RainbowSeason']:
            print(phase)
            task.set_phase(phase)
            agent.task_phase = phase
            n_trials = int(task.n_trials_per_phase[np.array(task.phases) == task.phase])
            for trial in range(n_trials):
                if interactive:
                    stimulus = sim_int.trial(trial)
                    [task.context, task.alien] = stimulus
                    sim_int.print_values_pre()
                    action = int(input('Action (0, 1, 2):'))
                else:
                    task.prepare_trial(trial)
                    stimulus = task.present_stimulus(trial)
                    action = agent.select_action(stimulus)
                [reward, correct] = task.produce_reward(action)
                agent.learn(stimulus, action, reward)
                if interactive:
                    sim_int.print_values_post(action, reward, correct)
                record_data.add_behavior(task, stimulus, action, reward, correct, total_trials, phase)
                record_data.add_decisions(agent, total_trials, suff='', all_Q_columns=all_Q_columns)
                total_trials += 1

        record_data.add_parameters(agent, '')  # add parameters (alpha, beta, etc.) only
        return record_data.get()

    def calculate_NLL(self, params_inf, agent_data, default_pars_lim="default", goal='calculate_NLL'):

        # Combine params_inf (current guess for fitted parameters) and default_pars_lim (default parameters)
        if default_pars_lim == "default":
            default_pars_lim = self.parameters.default_pars_lim
        pars_01 = self.parameters.change_limits(default_pars_lim, 'lim_to_01')
        pars_inf = self.parameters.change_scale(pars_01, '01_to_inf')
        fit_par_idx = np.argwhere(self.parameters.fit_pars)
        pars_inf[fit_par_idx] = params_inf

        # Bring parameters into the right scale and check that there was no error in conversion
        pars_01 = self.parameters.change_scale(pars_inf, 'inf_to_01')
        pars_lim = self.parameters.change_limits(pars_01, '01_to_lim')
        # for i, par in enumerate(self.parameters.par_names):
        #     assert((np.round(pars_lim[i], 2) == np.round(default_pars_lim[i], 2)) or self.parameters.fit_pars[i])

        # Create agent with these parameters
        agent = Agent(self.agent_stuff, pars_lim, self.task_stuff)
        if goal == 'add_decisions_and_fit':
            record_data = RecordData(agent_id=agent.id,
                                     mode='add_to_existing_data',
                                     agent_data=agent_data)

        # Let the agent do the task
        n_trials = len(agent_data)
        for trial in range(n_trials):
            if 'alien' in self.agent_stuff['name']:
                if np.isnan(agent_data['context'][trial]):
                    break
                context = int(agent_data['context'][trial])
                sad_alien = int(agent_data['sad_alien'][trial])
                stimulus = np.array([context, sad_alien])
                agent.select_action(stimulus)  # calculate p_actions
                action = int(agent_data['item_chosen'][trial])
            elif 'PS' in self.agent_stuff['name']:
                stimulus = np.nan
                agent.select_action(stimulus)  # calculate p_actions
                action = int(agent_data['selected_box'][trial])
            reward = int(agent_data['reward'][trial])
            agent.learn(stimulus, action, reward)
            if goal == 'add_decisions_and_fit':
                record_data.add_decisions(agent, trial, suff='_rec')

        BIC = - 2 * agent.LL + self.n_fit_par * np.log(n_trials)
        AIC = - 2 * agent.LL + self.n_fit_par

        if goal == 'calculate_NLL':
            return -agent.LL
        elif goal == 'calculate_fit':
            return [-agent.LL, BIC, AIC]
        elif goal == 'add_decisions_and_fit':
            record_data.add_parameters(agent, self.parameters, suff='_rec')  # add parameters and fit_pars
            record_data.add_fit(-agent.LL, BIC, AIC, suff='_rec')
            return record_data.get()

    def get_optimal_pars(self, agent_data, n_iter, default_pars_lim="default"):
        # x0 = brute(func=self.calculate_NLL,
        #            ranges=self.parameters.par_hard_limits,  # should be in the form (slice(-4, 4, 0.25), slice(-4, 4, 0.25))
        #            args=agent_data,
        #            full_output=True,
        #            finish=None)  # should try! https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brute.html#scipy.optimize.brute
        # optimized_params = self.parameters.inf_to_lim(x0)
        if default_pars_lim == "default":
            default_pars_lim = self.parameters.default_pars_lim

        # Find the minimum of n_iter different start values to get fit parameters
        values = np.zeros([n_iter, self.n_fit_par + 1])
        for iter in range(n_iter):
            start_par_inf = self.parameters.create_random_params(scale='inf', get_all=False)
            minimization = minimize(self.calculate_NLL,
                                    x0=start_par_inf,
                                    args=(agent_data, default_pars_lim),
                                    method='Nelder-Mead')
            values[iter, :] = np.concatenate(([minimization.fun], minimization.x))
        minimum = values[:, 0] == min(values[:, 0])
        fit_pars_inf = values[minimum][0]
        fit_pars_inf = fit_pars_inf[1:]

        # Combine fit parameters and fixed parameters and return all
        fixed_pars_01 = self.parameters.change_limits(default_pars_lim, 'lim_to_01')
        fixed_pars_inf = self.parameters.change_scale(fixed_pars_01, '01_to_inf')
        fit_par_idx = np.argwhere(self.parameters.fit_pars)
        fixed_pars_inf[fit_par_idx] = fit_pars_inf
        fixed_pars_01 = self.parameters.change_scale(fixed_pars_inf, 'inf_to_01')
        default_pars_lim = self.parameters.change_limits(fixed_pars_01, '01_to_lim')
        return default_pars_lim

    def write_agent_data(self, agent_data, save_path, file_name=''):
        agent_data.to_csv(save_path + file_name + "_" + str(agent_data['sID'][0]) + ".csv")
