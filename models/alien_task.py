import numpy as np
import scipy.io as spio


class Task(object):
    def __init__(self, task_stuff, agent_id):
        self.n_trials = task_stuff['n_trials']
        self.path = task_stuff['path']
        self.shuffled_aliens = np.random.choice(range(4), size=4, replace=False)
        self.context = np.nan
        self.alien = np.nan
        self.TS = np.array([[[0, 6, 0],  # TS0, alien0, items0-2
                             [0, 0, 4],  # TS0, alien1, items0-2
                             [5, 0, 0],  # etc.
                             [10, 0, 0]],
                            [[0, 0, 2],  # TS1, alien0, items0-2
                             [0, 8, 0],  # etc.
                             [0, 0, 7],
                             [0, 3, 0]],
                            [[0, 0, 7],  # TS2
                             [3, 0, 0],
                             [0, 3, 0],
                             [2, 0, 0]]])

        # self.TS = np.array([[1, 2, 0, 0],   # TS0
        #                     [2, 1, 2, 1],   # TS1
        #                     [2, 0, 1, 0]])  # TS2
        # self.rewards = np.array([[6, 4, 5, 10],  # TS0
        #                          [2, 8, 7, 3],   # TS1
        #                          [7, 3, 3, 2]])  # TS2

    def prepare_trial(self, trial):
        # Mix the order of the four aliens every 4 trials
        if trial % 4 == 0:
            self.shuffled_aliens = np.random.choice(range(4), size=4, replace=False)

    def present_stimulus(self, context, trial):
        # Return a season-alien stimulus
        self.context = context
        self.alien = self.shuffled_aliens[trial // 4]
        return np.array([self.context, self.alien])

    def produce_reward(self, action):
        # correct = self.TS[self.context, self.alien] == action
        # if correct:
            return self.rewards[self.context, self.alien, action]
        # else:
        #     return 0
