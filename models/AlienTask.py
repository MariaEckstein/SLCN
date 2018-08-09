import numpy as np
import pandas as pd


class Task(object):

    def __init__(self, n_subj):

        self.n_subj = np.nan
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

    def get_trial_sequence(self, file_path):

        agent_data = pd.read_csv(file_path)

        # Remove all rows that do not contain 1InitialLearning data (-> jsPysch format)
        agent_data = agent_data.rename(columns={'TS': 'context'})  # rename "TS" column to "context"
        context_names = [str(TS) for TS in range(3)]
        item_names = range(4)
        phases = ["1InitialLearning", "Refresher2", "Refresher3"]
        agent_data = agent_data.loc[(agent_data['context'].isin(context_names)) &
                                    (agent_data['item_chosen'].isin(item_names)) &
                                    (agent_data['phase'].isin(phases))]
        n_trials = agent_data.shape[0]

        # Read out sequence of seasons and aliens
        self.seasons = np.tile(agent_data["context"], self.n_subj).reshape([self.n_subj, n_trials]).astype(int).T
        self.aliens = np.tile(agent_data["sad_alien"], self.n_subj).reshape([self.n_subj, n_trials]).astype(int).T
        self.phase = agent_data["phase"]

        return n_trials

    def present_stimulus(self, trial):

        # Look up alien and context for current trial
        self.alien = self.aliens[trial, :]
        self.season = self.seasons[trial, :]

        # Look up alien and season for current trial
        return self.season, self.alien

    def produce_reward(self, action):

        # Look up reward in TS table, determine if response was correct
        reward = np.array([self.TS[self.season[subj], self.alien[subj], action[subj]] for subj in range(self.n_subj)])
        correct = reward > 1

        # Add noise to reward
        noised_reward = np.round(np.random.normal(reward, 0.05), 2)
        noised_reward[noised_reward < 0] = 0

        return noised_reward, correct
