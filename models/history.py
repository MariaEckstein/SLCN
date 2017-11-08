import numpy as np
import pandas as pd
import os


class History(object):
    def __init__(self, task, agent):
        self.agent_id = agent.id
        colnames = ['values_l', 'values_r', 'selected_box', 'reward', 'p_switch', 'correct_box', 'sID', 'RT']
        subj_file = np.zeros([task.n_trials, len(colnames)])
        self.subj_file = pd.DataFrame(data=subj_file, columns=colnames)
        self.data_path = "C:/Users/maria/MEGAsync/SLCNdata/" + agent.name

    def update(self, agent, task, action, reward, trial):
        self.subj_file['selected_box'][trial] = action
        self.subj_file['reward'][trial] = reward
        self.subj_file['correct_box'][trial] = task.correct_box
        if agent.name == 'RL':
            self.subj_file['values_l'][trial] = agent.q[0]
            self.subj_file['values_r'][trial] = agent.q[1]
        else:
            self.subj_file['values_l'][trial] = agent.p_boxes[0]
            self.subj_file['values_r'][trial] = agent.p_boxes[1]
            self.subj_file['p_switch'][trial] = agent.p_switch

    def save_csv(self, agent_id):
        self.subj_file['sID'] = self.agent_id
        self.subj_file['RT'] = np.nan
        if not os.path.isdir(self.data_path):
            os.makedirs(self.data_path)
        self.subj_file.to_csv(self.data_path + "/" + "/agent" + str(agent_id) + ".csv")
