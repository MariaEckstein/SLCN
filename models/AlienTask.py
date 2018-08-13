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

    def get_trial_sequence(self, file_paths):

        for file_path in file_paths:
            agent_data = pd.read_csv(file_path)

            # Remove all rows that do not contain 1InitialLearning data (-> jsPysch format)
            agent_data = agent_data.rename(columns={'TS': 'context'})  # rename "TS" column to "context"
            context_names = [str(TS) for TS in range(3)]
            phases = ["1InitialLearning", "Refresher2", "Refresher3"]
            agent_data = agent_data.loc[(agent_data['context'].isin(context_names)) &
                                        (agent_data['phase'].isin(phases))]

            # Read out sequence of seasons and aliens
            try:
                seasons = np.hstack([seasons, agent_data["context"]])
                aliens = np.hstack([aliens, agent_data["sad_alien"]])
                phase = np.hstack([phase, agent_data["phase"]])
            except:
                seasons = agent_data["context"]
                aliens = agent_data["sad_alien"]
                phase = agent_data["phase"]

        # Bring into right shape
        n_trials = agent_data.shape[0]
        seasons = seasons.reshape([len(file_paths), n_trials]).T
        aliens = aliens.reshape([len(file_paths), n_trials]).T

        # Read out sequence of seasons and aliens
        replic = int(np.ceil(self.n_subj / len(file_paths)))
        self.seasons = np.tile(seasons, replic).astype(int)
        self.aliens = np.tile(aliens, replic).astype(int)
        self.phase = agent_data["phase"]

        return n_trials

    def present_stimulus(self, trial):

        # Look up alien and context for current trial
        self.alien = self.aliens[trial, :self.n_subj]
        self.season = self.seasons[trial, :self.n_subj]

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
