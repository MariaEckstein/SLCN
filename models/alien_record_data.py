import numpy as np
import pandas as pd


class RecordData(object):

    def __init__(self, n_trials, agent_id, mode='add_to_existing_data', agent_data=()):
        if mode == 'create_from_scratch':
            colnames = ['trial_type', 'trial_index', 'correct', 'reward', 'item_chosen', 'sad_alien', 'TS', 'block-type']
            self.subj_file = pd.DataFrame(data=np.zeros([n_trials, len(colnames)]),
                                          columns=colnames)
            self.subj_file['sID'] = agent_id
            self.subj_file['RT'] = np.nan
        else:
            self.subj_file = agent_data

    def add_parameters(self, agent, suff=''):
        self.subj_file['alpha' + suff] = agent.alpha
        self.subj_file['beta' + suff] = agent.beta
        self.subj_file['epsilon' + suff] = agent.epsilon
        self.subj_file['perseverance' + suff] = agent.perseverance
        self.subj_file['decay' + suff] = agent.decay
        self.subj_file['mix' + suff] = agent.mix

    def add_behavior(self, task, stimulus, action, reward, trial, suff=''):
        self.subj_file.loc[trial, 'TS' + suff] = stimulus[0]
        self.subj_file.loc[trial, 'sad_alien' + suff] = stimulus[1]
        self.subj_file.loc[trial, 'item_chosen' + suff] = action
        self.subj_file.loc[trial, 'reward' + suff] = reward
        self.subj_file.loc[trial, 'trial_index'] = trial
        self.subj_file.loc[trial, 'correct'] = reward > 0

    def add_decisions(self, agent, trial, suff=''):
        self.subj_file.loc[trial, 'LL' + suff] = agent.LL
        [TS, alien] = self.subj_file.loc[trial, ['TS', 'sad_alien']]
        for i in range(3):
            self.subj_file.loc[trial, 'Q_low' + str(i) + suff] = agent.Q_low[TS, alien, int(i)]
            self.subj_file.loc[trial, 'Q_high' + str(i) + suff] = agent.Q_high[int(i)]
            self.subj_file.loc[trial, 'p_action' + str(i) + suff] = agent.p_actions[int(i)]

    def add_fit(self, NLL, BIC, AIC, suff=''):
        self.subj_file['NLL' + suff] = NLL
        self.subj_file['BIC' + suff] = BIC
        self.subj_file['AIC' + suff] = AIC

    def get(self):
        # self.subj_file.describe()
        return self.subj_file.copy()
