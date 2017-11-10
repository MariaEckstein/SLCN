import numpy as np
import pandas as pd
import os


class History(object):
    def __init__(self, task, agent, path):
        self.agent_id = agent.id
        self.params = [agent.alpha, agent.beta, agent.epsilon, agent.perseverance, agent.decay]
        self.LL = agent.LL
        colnames = ['LL', 'BIC', 'AIC', 'values_l', 'values_r', 'alpha', 'beta', 'epsilon', 'perseverance', 'decay',
                    'selected_box', 'reward', 'p_switch', 'correct_box', 'sID', 'RT']
        subj_file = np.zeros([task.n_trials, len(colnames)])
        self.subj_file = pd.DataFrame(data=subj_file, columns=colnames)
        self.data_path = "C:/Users/maria/MEGAsync/SLCNdata/" + agent.learning_style + '/' + agent.method + '/' + path

    def update(self, agent, task, action, reward, trial):
        self.subj_file['selected_box'][trial] = action
        self.subj_file['reward'][trial] = reward
        self.subj_file['correct_box'][trial] = task.correct_box
        self.subj_file['LL'][trial] = agent.LL
        if agent.learning_style == 'RL':
            self.subj_file[['values_l', 'values_r']][trial] = agent.q
        elif agent.learning_style == 'Bayes':
            self.subj_file[['values_l', 'values_r']][trial] = agent.p_boxes
            self.subj_file['p_switch'][trial] = agent.p_switch

    def save_csv(self):
        self.subj_file[['alpha', 'beta', 'epsilon', 'perseverance', 'decay']] = self.params
        self.subj_file['BIC'] = - 2 * self.LL + n_fit_params * np.log(task.n_trials)
        self.subj_file['sID'] = self.agent_id
        self.subj_file['RT'] = np.nan
        self.subj_file = self.subj_file.drop(0)
        if not os.path.isdir(self.data_path):
            os.makedirs(self.data_path)
        self.subj_file.to_csv(self.data_path + "/PS_" + str(self.agent_id) + ".csv")
