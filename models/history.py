import numpy as np
import pandas as pd
import os


class History(object):
    def __init__(self, task, agent, path):
        self.agent_id = agent.id
        # self.LL = agent.LL
        # self.n_free_par = sum(agent.free_par)
        # self.n_actions = agent.n_actions
        colnames = ['LL', 'BIC', 'AIC', 'values_l', 'values_r', 'alpha', 'beta', 'epsilon', 'perseverance', 'decay',
                    'selected_box', 'reward', 'p_switch', 'correct_box', 'sID', 'RT']
        subj_file = np.zeros([task.n_trials, len(colnames)])
        self.subj_file = pd.DataFrame(data=subj_file, columns=colnames)
        self.subj_file[['alpha', 'beta', 'epsilon', 'perseverance', 'decay']] =\
            [agent.alpha, agent.beta, agent.epsilon, agent.perseverance, agent.decay]
        self.data_path = agent.data_path + '/' + path

    def update(self, agent, task, action, reward, trial):
        self.subj_file.loc[trial, 'selected_box'] = action
        self.subj_file.loc[trial, 'reward'] = reward
        self.subj_file.loc[trial, 'correct_box'] = task.correct_box
        self.subj_file.loc[trial, 'LL'] = agent.LL
        if agent.learning_style == 'RL':
            self.subj_file.loc[trial, ['values_l', 'values_r']] = agent.q
        elif agent.learning_style == 'Bayes':
            self.subj_file.loc[trial, ['values_l', 'values_r']] = agent.p_boxes
            self.subj_file.loc[trial, 'p_switch'] = agent.p_switch

    def save_csv(self):
        # self.subj_file['BIC'] = - 2 * self.LL + self.n_free_par * np.log(self.n_actions)
        # self.subj_file['AIC'] = - 2 * self.LL + self.n_free_par
        self.subj_file['sID'] = self.agent_id
        self.subj_file['RT'] = np.nan
        self.subj_file = self.subj_file.drop(0)
        if not os.path.isdir(self.data_path):
            os.makedirs(self.data_path)
        self.subj_file.to_csv(self.data_path + "/PS_" + str(self.agent_id) + ".csv")
