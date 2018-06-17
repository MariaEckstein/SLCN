import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import brute
from scipy.optimize import basinhopping
from BasinhoppinTakeStep import MyTakeStep
from BasinhoppinBounds import MyBounds
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
        self.n_fit_par = sum(parameters['fit_pars'])

    # def get_agent_data(self, way, data_path='', all_Q_columns=False, all_pars=()):
    #     assert(way in ['simulate', 'real_data', 'interactive'])
    #     if way in ['simulate', 'interactive']:
    #         return self.simulate_agent(all_pars, all_Q_columns, interactive=way == 'interactive')
    #     elif way == 'real_data':
    #         return pd.read_csv(data_path)

    def simulate_agent(self, all_pars, all_Q_columns, interactive=False):

        task = Task(self.task_stuff)
        agent = Agent(self.agent_stuff, all_pars, self.task_stuff)
        record_data = RecordData(agent_id=agent.id,
                                 mode='create_from_scratch',
                                 task=task)
        if interactive:
            sim_int = SimulateInteractive(agent, self.agent_stuff['mix_probs'])

        total_trials = 0
        for phase in ['1InitialLearning', '2CloudySeason', 'Refresher2']:
            task.set_phase(phase)
            agent.task_phase = phase
            n_trials = int(task.n_trials_per_phase[np.array(task.phases) == task.phase])
            for trial in range(n_trials):
                if trial == 0 and phase == '2CloudySeason':
                    agent.prev_context = 99
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

    def calculate_NLL(self, vary_pars, agent_data, default_pars="default", goal='calculate_NLL', suff='_rec'):

        # Combine params_inf (current guess for fitted parameters) and default_pars (default parameters)
        if default_pars == "default":
            default_pars = self.parameters['default_pars']
        default_pars[np.argwhere(self.parameters['fit_pars'])] = vary_pars

        # Create agent with these parameters
        agent = Agent(self.agent_stuff, default_pars, self.task_stuff)
        if goal == 'add_decisions_and_fit':
            record_data = RecordData(agent_id=agent.id,
                                     mode='add_to_existing_data',
                                     agent_data=agent_data)

        # Let the agent do the task
        n_trials = len(agent_data)
        for trial in range(n_trials):
            if 'alien' in self.agent_stuff['name']:
                agent.task_phase = '1InitialLearning'
                context = int(agent_data['context'][trial])
                sad_alien = int(agent_data['sad_alien'][trial])
                stimulus = np.array([context, sad_alien])
                agent.select_action(stimulus)  # calculate p_actions
                action = int(float(agent_data['item_chosen'][trial]))
            elif 'PS' in self.agent_stuff['name']:
                stimulus = np.nan
                agent.select_action(stimulus)  # calculate p_actions
                action = int(agent_data['selected_box'][trial])
            reward = int(agent_data['reward'][trial])
            agent.learn(stimulus, action, reward)
            if goal == 'add_decisions_and_fit':
                record_data.add_decisions(agent, trial, suff=suff)

        BIC = - 2 * agent.LL + self.n_fit_par * np.log(n_trials)
        AIC = - 2 * agent.LL + self.n_fit_par

        if goal == 'calculate_NLL':
            # print(-agent.LL, vary_pars)
            return -agent.LL
        elif goal == 'calculate_fit':
            return [-agent.LL, BIC, AIC]
        elif goal == 'add_decisions_and_fit':
            record_data.add_parameters(agent, self.parameters, suff=suff)  # add parameters and fit_pars
            record_data.add_fit(-agent.LL, BIC, AIC, suff=suff)
            return record_data.get()

    def get_optimal_pars(self, agent_data, n_iter, default_pars="default"):
        if default_pars == "default":
            default_pars = self.parameters['default_pars']

        # Calculate the minimum value by brute force
        minimization = brute(func=self.calculate_NLL,
                             ranges=([self.parameters['par_hard_limits'][i] for i in np.argwhere(self.parameters['fit_pars'])]),
                             args=(agent_data, default_pars),
                             Ns=n_iter,  # n_fit_par-te Wurzel aus (n_iter * 1000) -> gleich viel effort wie unten
                             full_output=True,
                             finish=None,
                             disp=True)
        start_par = minimization[0]  # np.array([0.1, 0.1]),  #
        print("Finished brute with values {0}!".format(str(np.round(start_par, 3))))

        takestep = MyTakeStep(limits=(0, 1), stepsize=0.5)
        bounds = MyBounds(limits=(0, 1))
        minimization = basinhopping(func=self.calculate_NLL,
                                    x0=start_par,
                                    niter=n_iter,  # The number of basin hopping iterations
                                    T=50.0,  # For best results T should be comparable to the separation (in function value) between local minima. it seems like NLLs are easily off by 100, telling from gg_gen_rec
                                    # stepsize=0.5,  # initial step size for use in the random displacement. Ideally it should be comparable to the typical separation between local minima of the function being optimized. gg_gen_rec implies that alpha is often off by 0.25; beta by -5
                                    take_step=takestep,  # never take a step outside of the limits
                                    accept_test=bounds,  # Define a test which will be used to judge whether or not to accept the step. This will be used in addition to the Metropolis test based on “temperature” T.
                                    minimizer_kwargs={'method': 'Nelder-Mead',  # options: Nelder-Mead, Powell, CG, BFGS
                                                      'args': (agent_data, default_pars),
                                                      'options': {'xatol': .01,  # 'gtol': .001, #
                                                                  'fatol': .01,
                                                                  'maxfev': 300}},  # 'maxiter': 100}},  #
                                    disp=True)
        fit_pars_lim = minimization.x

        # Combine fit parameters and fixed parameters and return all
        fit_par_idx = np.argwhere(self.parameters['fit_pars'])
        default_pars[fit_par_idx] = fit_pars_lim
        return default_pars

    # def write_agent_data(self, agent_data, save_path, file_name=''):
    #     agent_data.to_csv(save_path + file_name + "_" + str(agent_data['sID'][0]) + ".csv")
