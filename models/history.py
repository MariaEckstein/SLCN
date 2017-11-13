import numpy as np
import pandas as pd
import os


class History(object):
    def __init__(self, task, agent):
        self.agent_id = agent.id
        colnames = ['NLL', 'BIC', 'AIC', 'LL', 'values_l', 'values_r', 'p_action_l', 'p_action_r',
                    'alpha', 'beta', 'epsilon', 'perseverance', 'decay',
                    'selected_box', 'reward', 'p_switch', 'correct_box', 'sID', 'RT']
        subj_file = np.zeros([task.n_trials, len(colnames)])
        self.subj_file = pd.DataFrame(data=subj_file, columns=colnames)
        self.subj_file[['alpha', 'beta', 'epsilon', 'perseverance', 'decay']] =\
            [agent.alpha, agent.beta, agent.epsilon, agent.perseverance, agent.decay]
        self.data_path = agent.data_path

    def update(self, agent, task, action, reward, trial):
        self.subj_file.loc[trial, 'selected_box'] = action
        self.subj_file.loc[trial, 'reward'] = reward
        self.subj_file.loc[trial, 'correct_box'] = task.correct_box
        self.subj_file.loc[trial, 'LL'] = agent.LL
        self.subj_file.loc[trial, ['p_action_l', 'p_action_r']] = agent.p_actions
        if agent.learning_style == 'RL':
            self.subj_file.loc[trial, ['values_l', 'values_r']] = agent.q
        elif agent.learning_style == 'Bayes':
            self.subj_file.loc[trial, ['values_l', 'values_r']] = agent.p_actions
            self.subj_file.loc[trial, 'p_switch'] = agent.p_switch

    def save_csv(self, fit):
        self.subj_file[['NLL', 'BIC', 'AIC']] = fit
        self.subj_file['sID'] = self.agent_id
        self.subj_file['RT'] = np.nan
        self.subj_file = self.subj_file.drop(0)
        if not os.path.isdir(self.data_path):
            os.makedirs(self.data_path)
        self.subj_file.to_csv(self.data_path + "/PS_" + str(self.agent_id) + ".csv")
