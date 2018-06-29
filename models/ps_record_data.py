import numpy as np
import pandas as pd


class RecordData(object):

    def __init__(self, mode='add_to_existing_data', agent_data=(), task=()):
        if mode == 'create_from_scratch':
            colnames = ['selected_box', 'reward', 'sID', 'RT']
            self.subj_file = pd.DataFrame(data=np.zeros([task.n_trials, len(colnames)]),
                                          columns=colnames)
            self.subj_file['RT'] = np.nan
        else:
            self.subj_file = agent_data

    def add_parameters(self, agent, agent_id, parameters=None, suff=''):
        self.subj_file['learning_style'] = agent.learning_style
        if agent_id is not None:
            self.subj_file['sID'] = agent_id
        if parameters:
            self.subj_file['fit_pars' + suff] = str(parameters['fit_pars'])
        if hasattr(agent, 'alpha'):
            self.subj_file['alpha' + suff] = agent.alpha
            self.subj_file['alpha_high' + suff] = agent.alpha_high
            self.subj_file['beta' + suff] = agent.beta
            self.subj_file['beta_high' + suff] = agent.beta_high
        if hasattr(agent, 'epsilon'):
            self.subj_file['epsilon' + suff] = agent.epsilon
        if hasattr(agent, 'forget'):
            self.subj_file['forget' + suff] = agent.forget

    def add_behavior(self, action, reward, correct, correct_box, trial, suff=''):
        self.subj_file.loc[trial, 'selected_box' + suff] = action
        self.subj_file.loc[trial, 'reward' + suff] = reward
        self.subj_file.loc[trial, 'correct' + suff] = correct
        self.subj_file.loc[trial, 'correct_box' + suff] = correct_box

    def add_decisions(self, agent, trial, suff=''):
        self.subj_file.loc[trial, 'LL' + suff] = agent.LL
        self.subj_file.loc[trial, 'p_action_l' + suff] = agent.p_actions[0]
        self.subj_file.loc[trial, 'p_action_r' + suff] = agent.p_actions[1]
        if hasattr(agent, 'Q_low'):
            self.subj_file.loc[trial, 'q_action_l' + suff] = agent.Q_low[0, 0]  # TS0, action0
            self.subj_file.loc[trial, 'q_action_l' + suff] = agent.Q_low[0, 1]  # TS0, action1

    def add_fit(self, NLL, BIC, AIC, suff=''):
        self.subj_file['NLL' + suff] = NLL
        self.subj_file['BIC' + suff] = BIC
        self.subj_file['AIC' + suff] = AIC

    def get(self):
        # self.subj_file.describe()
        return self.subj_file.copy()
