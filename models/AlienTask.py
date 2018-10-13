import numpy as np
import pandas as pd
import glob
import os


class Task(object):

    def __init__(self, n_subj):

        self.n_subj = n_subj
                            # TS0
        self.TS = np.array([[[1, 6, 1],  # alien0, items0-2
                             [1, 1, 4],  # alien1, items0-2
                             [5, 1, 1],  # etc.
                             [10, 1, 1]],
                            # TS1
                            [[1, 1, 2],  # alien0, items0-2
                             [1, 8, 1],  # etc.
                             [1, 1, 7],
                             [1, 3, 1]],
                            # TS2
                            [[1, 1, 7],  # TS2
                             [3, 1, 1],
                             [1, 3, 1],
                             [2, 1, 1]]])

    def get_trial_sequence(self, file_path, n_subj, n_sim_per_subj, fake=False):
        '''
        Get trial sequences of human participants.
        Read in datafiles of all participants, select InitialLearning, Refresher2, and Refresher3,
        and get seasons and aliens in each trial.
        :param file_path: path to human datafiles
        :param n_subj: number of files to be read in
        :return: n_trials: number of trials in each file
        '''

        filenames = glob.glob(os.path.join(file_path, '*.csv'))
        n_trials = 100000
        for filename in filenames[:n_subj]:
            agent_data = pd.read_csv(filename)

            # Remove all rows that do not contain 1InitialLearning data (-> jsPysch format)
            TS_names = [str(TS) for TS in range(3)]
            phases = ["1InitialLearning", "Refresher2", "Refresher3"]
            agent_data = agent_data.loc[(agent_data['TS'].isin(TS_names)) &
                                        (agent_data['phase'].isin(phases))]

            # Read out sequence of seasons and aliens
            try:
                seasons = np.hstack([seasons, agent_data["TS"]])
                aliens = np.hstack([aliens, agent_data["sad_alien"]])
                phase = np.hstack([phase, agent_data["phase"]])
                correct = np.hstack([correct, agent_data["correct"]])
            except NameError:
                seasons = agent_data["TS"]
                aliens = agent_data["sad_alien"]
                phase = agent_data["phase"]
                correct = agent_data["correct"]
            n_trials = np.min([n_trials, agent_data.shape[0]])

        # Bring into right shape
        seasons = np.tile(seasons, n_sim_per_subj)
        aliens = np.tile(aliens, n_sim_per_subj)
        self.seasons = seasons.reshape([n_subj * n_sim_per_subj, n_trials]).astype(int).T
        self.aliens = aliens.reshape([n_subj * n_sim_per_subj, n_trials]).astype(int).T
        self.phase = np.tile(agent_data["phase"], n_sim_per_subj)

        if fake:
            self.seasons = np.tile(np.tile(np.repeat(np.arange(3), 80), 4), n_subj).reshape([n_subj, 3 * 80 * 4]).T  # np.zeros([n_subj, n_trials], dtype=int).T
            self.aliens = np.tile(np.arange(4), int(n_subj * 80 * 3)).reshape([n_subj, 3 * 80 * 4]).astype(int).T
            n_trials = self.seasons.shape[0]
            self.phase = '1InitialLearning'

        return n_trials, correct.reshape([n_subj, n_trials]).astype(int).T

    def present_stimulus(self, trial):

        # Look up alien and context for current trial
        self.alien = self.aliens[trial]  # , :self.n_subj * ]
        self.season = self.seasons[trial]  # , :self.n_subj]

        return self.season, self.alien

    def produce_reward(self, action):

        # Look up reward in TS table, determine if response was correct
        reward = self.TS[self.season, self.alien, action]
        correct = reward > 1

        # Add noise to reward
        noised_reward = np.round(np.random.normal(reward, 0.05), 2)
        noised_reward[noised_reward < 0] = 0

        return noised_reward, correct
