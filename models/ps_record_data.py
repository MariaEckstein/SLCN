import numpy as np
import pandas as pd


class RecordData(object):

    def __init__(self, n_trials, agent_id, mode='add_to_existing_data', agent_data=()):
        if mode == 'create_from_scratch':
            colnames = ['selected_box', 'reward', 'p_switch', 'correct_box', 'sID', 'RT']
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
        self.subj_file['w_reward' + suff] = agent.w_reward
        self.subj_file['w_noreward' + suff] = agent.w_noreward
        self.subj_file['w_explore' + suff] = agent.w_explore

    def add_behavior(self, task, stimulus, action, reward, trial, suff=''):
        self.subj_file.loc[trial, 'selected_box' + suff] = action
        self.subj_file.loc[trial, 'reward' + suff] = reward
        self.subj_file.loc[trial, 'correct_box' + suff] = task.correct_box

    def add_decisions(self, agent, trial, suff=''):
        self.subj_file.loc[trial, 'LL' + suff] = agent.LL
        self.subj_file.loc[trial, 'p_action_l' + suff] = agent.p_actions[0]
        self.subj_file.loc[trial, 'p_action_r' + suff] = agent.p_actions[1]
        if agent.learning_style == 'RL':
            self.subj_file.loc[trial, 'values_l' + suff] = agent.q[0]
            self.subj_file.loc[trial, 'values_r' + suff] = agent.q[1]
        else:  # if agent.learning_style == 'Bayes' or 'estimate-switch'
            self.subj_file.loc[trial, 'values_l' + suff] = agent.p_actions[0]
            self.subj_file.loc[trial, 'values_r' + suff] = agent.p_actions[1]
            self.subj_file.loc[trial, 'p_switch' + suff] = agent.p_switch

    def add_fit(self, NLL, BIC, AIC, suff=''):
        self.subj_file['NLL' + suff] = NLL
        self.subj_file['BIC' + suff] = BIC
        self.subj_file['AIC' + suff] = AIC

    def get(self):
        # self.subj_file.describe()
        return self.subj_file.copy()
