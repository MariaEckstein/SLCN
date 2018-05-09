import numpy as np
import pandas as pd


class RecordData(object):

    def __init__(self, agent_id, mode='add_to_existing_data', agent_data=(), task=()):
        if mode == 'create_from_scratch':
            colnames = ['trial_type', 'trial_index', 'correct', 'reward', 'item_chosen', 'sad_alien', 'TS', 'block-type']
            self.subj_file = pd.DataFrame(data=np.zeros([task.n_trials, len(colnames)]),
                                          columns=colnames)
            self.subj_file['n_trials_per_alien'] = task.n_trials_per_alien
            self.subj_file['n_blocks'] = task.n_blocks
            self.subj_file['sID'] = agent_id
            self.subj_file['RT'] = np.nan
        else:
            self.subj_file = agent_data

    def add_parameters(self, agent, parameters, suff=''):
        if parameters:
            self.subj_file['fit_pars'] = str(parameters.fit_pars)
        self.subj_file['n_actions'] = agent.n_actions
        self.subj_file['n_TS'] = agent.n_TS
        self.subj_file['method'] = agent.method
        self.subj_file['learning_style'] = agent.learning_style
        self.subj_file['mix_probs'] = agent.mix_probs
        self.subj_file['alpha' + suff] = agent.alpha
        self.subj_file['alpha_high' + suff] = agent.alpha_high
        self.subj_file['beta' + suff] = agent.beta
        self.subj_file['epsilon' + suff] = agent.epsilon
        # self.subj_file['perseverance' + suff] = agent.perseverance
        # self.subj_file['forget' + suff] = agent.forget
        # self.subj_file['forget_high' + suff] = agent.forget_high
        # self.subj_file['mix' + suff] = agent.mix

    def add_behavior(self, task, stimulus, action, reward, correct, trial, suff=''):
        self.subj_file.loc[trial, 'context' + suff] = stimulus[0]
        self.subj_file.loc[trial, 'sad_alien' + suff] = stimulus[1]
        self.subj_file.loc[trial, 'item_chosen' + suff] = action
        self.subj_file.loc[trial, 'reward' + suff] = reward
        self.subj_file.loc[trial, 'correct'] = correct
        self.subj_file.loc[trial, 'trial_index'] = trial

    def add_decisions(self, agent, trial, suff='', all_Q_columns=False):
        current_context = int(self.subj_file.loc[trial, 'context'])
        current_TS = int(agent.TS)
        current_alien = int(self.subj_file.loc[trial, 'sad_alien'])
        current_item_chosen = int(self.subj_file.loc[trial, 'item_chosen'])
        self.subj_file.loc[trial, 'LL' + suff] = agent.LL
        self.subj_file.loc[trial, 'TS' + suff] = current_TS
        # Q_high and Q_low for the current trial's TS and chosen item, respectively
        self.subj_file.loc[trial, 'Q_TS' + suff] = agent.Q_high[current_context, current_TS]
        self.subj_file.loc[trial, 'Q_action' + suff] = agent.Q_low[current_TS, current_alien, current_item_chosen]
        # p_high and p_low for the current trial's TS and chosen item, respectively
        self.subj_file.loc[trial, 'p_TS' + suff] = agent.p_TS[current_TS]
        self.subj_file.loc[trial, 'p_action' + suff] = agent.p_actions[current_item_chosen]
        # Add all values
        if all_Q_columns:
            for TS in range(3):
                self.subj_file.loc[trial, 'p_TS' + str(TS) + suff] = agent.p_TS[int(TS)]
                for context in range(3):
                    self.subj_file.loc[trial, 'Q_TS{0}_context{1}{2}'.format(str(TS), str(context), suff)] = agent.Q_high[int(context), TS]
            for action in range(3):
                self.subj_file.loc[trial, 'p_action' + str(action) + suff] = agent.p_actions[int(action)]
                for alien in range(4):
                    for TS in range(3):
                        self.subj_file.loc[trial, 'Q_action{0}_alien{1}_TS{2}{3}'.format(str(action), str(alien), str(TS), suff)]\
                            = agent.Q_low[action, alien, int(TS)]

    def add_fit(self, NLL, BIC, AIC, suff=''):
        self.subj_file['NLL' + suff] = NLL
        self.subj_file['BIC' + suff] = BIC
        self.subj_file['AIC' + suff] = AIC

    def get(self):
        # self.subj_file.describe()
        return self.subj_file.copy()
