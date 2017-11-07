import numpy as np
import pandas as pd
import os


class History(object):
    def __init__(self, task, n_trials, model_type):
        self.correct_box = np.zeros(n_trials)
        self.actions = np.zeros(n_trials)
        self.reward = np.full(n_trials, np.nan)
        self.values = np.zeros([n_trials, task.n_actions])
        self.switch_trial = np.zeros(n_trials)
        self.switch_prob = np.zeros(n_trials)
        colnames = ['ACC', 'values_l', 'values_r', 'key', 'reward', 'switch_prob', 'switch_trial', 'better_box_left', 'sID', 'rewardversion', 'RT']
        subj_file = np.zeros([n_trials, len(colnames)])
        self.subj_file = pd.DataFrame(data=subj_file, columns=colnames)
        self.data_path = "C:/Users/maria/MEGAsync/SLCNdata/" + model_type

    def update(self, agent, task, action, reward, switch, trial):
        self.correct_box[trial] = task.correct_box
        self.actions[trial] = action
        self.reward[trial] = reward
        self.switch_trial[trial] = switch
        if agent.name == 'RL':
            self.values[trial] = agent.q
        else:
            self.values[trial] = agent.p_boxes
            self.switch_prob[trial] = agent.switch_prob

    def transform_into_human_format(self, agent_id):
        self.subj_file['ACC'] = self.actions == self.correct_box  # not exactly true
        self.subj_file['values_l'] = self.values[:, 0]
        self.subj_file['values_r'] = self.values[:, 1]
        self.subj_file['key'] = self.actions
        self.subj_file['reward'] = self.reward
        self.subj_file['switch_prob'] = self.switch_prob
        self.subj_file['switch_trial'] = self.switch_trial
        self.subj_file['better_box_left'] = 1 - self.correct_box
        self.subj_file['sID'] = agent_id
        self.subj_file['rewardversion'] = np.nan
        self.subj_file['RT'] = np.nan

    def save_csv(self, agent_id):
        if not os.path.isdir(self.data_path):
            os.makedirs(self.data_path)
        self.subj_file.to_csv(self.data_path + "/" + "/agent" + str(agent_id) + ".csv")
