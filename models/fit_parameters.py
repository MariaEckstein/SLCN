import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import brute
from scipy.optimize import basinhopping
from basinhopping_specifics import AlienTakeStep, AlienBounds
from minimizer_heatmap import PlotMinimizerHeatmap, CollectPaths, CollectMinima
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

    def simulate_agent(self, all_pars, interactive=False):

        # Initialize task, agent, record, and interactive game
        task = Task(self.task_stuff)
        agent = Agent(self.agent_stuff, all_pars, self.task_stuff)
        record_data = RecordData(mode='create_from_scratch', task=task)
        if interactive:
            sim_int = SimulateInteractive(agent, self.agent_stuff['mix_probs'])

        # Play the game, phase by phase, trial by trial
        total_trials = 0
        for phase in ['1InitialLearning', '2CloudySeason', 'Refresher2']:
            task.set_phase(phase)
            agent.task_phase = phase
            n_trials = int(task.n_trials_per_phase[np.array(task.phases) == task.phase])
            agent.prev_context = 99
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
                record_data.add_decisions(agent, total_trials, suff='')
                total_trials += 1

        # task.set_phase('3PickAliens')
        # comp = CompetitionPhase(self.comp_stuff, self.task_stuff)
        # for trial in range(sum(comp.n_trials)):
        #     comp.prepare_trial(trial)
        #     stimuli = comp.present_stimulus(trial)
        #     selected = agent.competition_selection(stimuli, comp.current_phase)
        #     if interactive:
        #         print('\tTRIAL {0} ({1}),\nstimuli {2}, values: {3}, probs.: {4}'.format(
        #         trial, comp.current_phase, stimuli, str(np.round(agent.Q_stimuli, 2)), str(np.round(agent.p_stimuli, 2))))
        #     record_data.add_behavior_and_decisions_comp(stimuli, selected, agent.Q_stimuli, agent.p_stimuli,
        #                                                 total_trials, task.phase, comp.current_phase)
        #     total_trials += 1
        #
        # for phase in ['Refresher3', '5RainbowSeason']:
        #     task.set_phase(phase)
        #     agent.task_phase = phase
        #     n_trials = int(task.n_trials_per_phase[np.array(task.phases) == task.phase])
        #     for trial in range(n_trials):
        #         if interactive:
        #             stimulus = sim_int.trial(trial)
        #             [task.context, task.alien] = stimulus
        #             sim_int.print_values_pre()
        #             action = int(input('Action (0, 1, 2):'))
        #         else:
        #             task.prepare_trial(trial)
        #             stimulus = task.present_stimulus(trial)
        #             action = agent.select_action(stimulus)
        #         [reward, correct] = task.produce_reward(action)
        #         agent.learn(stimulus, action, reward)
        #         if interactive:
        #             sim_int.print_values_post(action, reward, correct)
        #         record_data.add_behavior(task, stimulus, action, reward, correct, total_trials, phase)
        #         record_data.add_decisions(agent, total_trials, suff='', all_Q_columns=all_Q_columns)
        #         total_trials += 1

        record_data.add_parameters(agent, '')  # add parameters (alpha, beta, etc.) only
        return record_data.get()

    def calculate_NLL(self, vary_pars, agent_data, collect_paths=None, verbose=False,
                      goal='calculate_NLL', suff='_rec'):

        # Get agent parameters
        all_pars = self.parameters['default_pars']
        all_pars[np.argwhere(self.parameters['fit_pars'])] = vary_pars

        # Initialize agent and record
        agent = Agent(self.agent_stuff, all_pars, self.task_stuff)
        if goal == 'add_decisions_and_fit':
            record_data = RecordData(mode='add_to_existing_data',
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

        # Calculate fit of this set of parameters
        BIC = - 2 * agent.LL + self.n_fit_par * np.log(n_trials)
        AIC = - 2 * agent.LL + self.n_fit_par

        if goal == 'calculate_NLL':
            if verbose:
                print(-agent.LL, vary_pars)
            if collect_paths:
                collect_paths.add_point(np.array(vary_pars))
            return -agent.LL
        elif goal == 'calculate_fit':
            return [-agent.LL, BIC, AIC]
        elif goal == 'add_decisions_and_fit':
            record_data.add_parameters(agent, self.parameters, suff=suff)  # add parameters and fit_pars
            record_data.add_fit(-agent.LL, BIC, AIC, suff=suff)
            return record_data.get()

    def get_optimal_pars(self, agent_data, minimizer_stuff):

        if minimizer_stuff['save_plot_data']:
            file_name = minimizer_stuff['plot_save_path'] + '/ID' + str(agent_data.loc[0, 'sID'])
            plot_heatmap = PlotMinimizerHeatmap(file_name)
            hoppin_paths = CollectPaths(colnames=self.parameters['fit_par_names'])
            brute_results = brute(func=self.calculate_NLL,
                                  ranges=([self.parameters['par_hard_limits'][i] for i in np.argwhere(self.parameters['fit_pars'])]),
                                  args=(agent_data, hoppin_paths, minimizer_stuff['verbose']),
                                  Ns=minimizer_stuff['brute_Ns'],
                                  full_output=True,
                                  finish=None,
                                  disp=True)
            print('Finished brute!')
            plot_heatmap.pickle_brute_results(brute_results)
            hoppin_minima = CollectMinima(colnames=self.parameters['fit_par_names'])
            hoppin_paths = CollectPaths(colnames=self.parameters['fit_par_names'])  # reinitialize
        else:
            hoppin_minima = None
            hoppin_paths = None

        n_free_pars = np.sum(self.parameters['fit_pars'])
        bounds = AlienBounds(xmax=np.ones(n_free_pars), xmin=np.zeros(n_free_pars))
        takestep = AlienTakeStep(stepsize=minimizer_stuff['hoppin_stepsize'],
                                 bounds=self.parameters['par_hard_limits'][0])
        hoppin_results = basinhopping(func=self.calculate_NLL,
                                      x0=.5 * np.ones(n_free_pars),
                                      niter=minimizer_stuff['NM_niter'],
                                      T=minimizer_stuff['hoppin_T'],
                                      minimizer_kwargs={'method': 'Nelder-Mead',
                                                        'args': (agent_data, hoppin_paths, minimizer_stuff['verbose']),
                                                        'options': {'xatol': minimizer_stuff['NM_xatol'],
                                                                    'fatol': minimizer_stuff['NM_fatol'],
                                                                    'maxfev': minimizer_stuff['NM_maxfev']}},
                                      take_step=takestep,
                                      accept_test=bounds,
                                      callback=hoppin_minima,
                                      disp=True)
        hoppin_fit_par, hoppin_NLL = [hoppin_results.x, hoppin_results.fun]

        if minimizer_stuff['save_plot_data']:
            fin_res = np.append(hoppin_fit_par, hoppin_NLL) * np.ones((1, 4))
            final_result = pd.DataFrame(fin_res, columns=self.parameters['fit_par_names'] + ['NLL'])
            final_result.to_csv(plot_heatmap.file_path + 'hoppin_result.csv')
            hoppin_paths.get().to_csv(plot_heatmap.file_path + 'hoppin_paths.csv')
            hoppin_minima.get().to_csv(plot_heatmap.file_path + 'hoppin_minima.csv')

        print("Finished basin hopping with values {0}, NLL {1}."
              .format(np.round(hoppin_fit_par, 3), np.round(hoppin_NLL, 3)))

        # Combine fit parameters and fixed parameters and return all
        fit_par_idx = np.argwhere(self.parameters['fit_pars'])
        minimized_pars = self.parameters['default_pars']
        minimized_pars[fit_par_idx] = hoppin_fit_par
        return minimized_pars
