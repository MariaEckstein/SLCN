import numpy as np
import scipy.io as spio


class Task(object):
    def __init__(self, info_path, n_subj):

        # Initialize counters for each subject
        self.n_subj = n_subj
        self.correct_box = np.random.binomial(1, 0.5, n_subj)
        self.n_rewards = np.zeros(n_subj, dtype=int)
        self.n_correct = np.zeros(n_subj, dtype=int)
        self.i_episode = np.zeros(n_subj, dtype=int)
        self.switched = np.zeros(n_subj, dtype=bool)

        # Load task specifics
        self.info_path = info_path  # path where randomization versions are stored
        n_runs = 200
        reward_versions = [str(i % 4) for i in range(n_subj)]
        self.run_lengths = np.array([spio.loadmat(self.info_path + '/run_length' + version + '.mat',
                                                  squeeze_me=True)['run_length'][:n_runs]
                                     for version in reward_versions]).T
        self.coin_wins = np.array([spio.loadmat(self.info_path + '/coin_win' + version + '.mat',
                                                squeeze_me=True)['coin_win'][:n_runs]
                                   for version in reward_versions], dtype=bool).T

    def prepare_trial(self):

        # Switch sides if necessary
        current_run_lengths = np.array([self.run_lengths[self.i_episode[subj], subj] for subj in range(self.n_subj)])
        switch = self.n_rewards == current_run_lengths
        self.correct_box[switch] = 1 - self.correct_box[switch]

        # Keep track of things
        self.switched[switch] = True  # for which subjects has the box switched sides?
        self.n_rewards[switch] = 0
        self.i_episode[switch] += 1

    def produce_reward(self, action):

        # Determine which subjects get coins
        current_coin_wins = np.array([self.coin_wins[self.n_correct[subj], subj] for subj in range(self.n_subj)])
        current_coin_wins[self.switched] = True  # always a coin when sides just switched (and correct action)
        current_coin_wins[action != self.correct_box] = False  # never a coin for incorrect action

        # Keep track of things
        self.switched[current_coin_wins] = False  # reset switch for subjects who got their coin after the last switch
        self.n_rewards[current_coin_wins] += 1
        self.n_correct[action == self.correct_box] += 1

        return current_coin_wins.astype(int)

    # def _exchange_rewards(self):
    #
    #     # Exchange rewards if necessary
    #     reward_indexes = np.argwhere(self.coin_win == 1)
    #     next_reward_index = reward_indexes[reward_indexes > self.n_correct][0]
    #     self.coin_win[next_reward_index] = 0
