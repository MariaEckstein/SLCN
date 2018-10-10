import numpy as np
import scipy.io as spio


class Task(object):
    def __init__(self, info_path, n_subj, subj_ids=None):

        # Initialize counters for each subject
        self.n_subj = n_subj
        if np.any(subj_ids):
            self.subj_ids = np.array(subj_ids, dtype=int)
        else:
            self.subj_ids = range(n_subj)
        print(self.subj_ids)
        self.correct_box = np.random.binomial(1, 0.5, n_subj)
        self.n_rewards = np.zeros(n_subj, dtype=int)
        self.n_correct = np.zeros(n_subj, dtype=int)
        self.i_episode = np.zeros(n_subj, dtype=int)
        self.switched = np.zeros(n_subj, dtype=bool)

        # Load task specifics
        self.info_path = info_path  # path where randomization versions are stored
        n_runs = 200
        reward_versions = [str(i % 4) for i in self.subj_ids]
        self.run_lengths = np.array([spio.loadmat(self.info_path + '/run_length' + version + '.mat',
                                                  squeeze_me=True)['run_length'][:n_runs]
                                     for version in reward_versions]).T
        self.coin_wins = np.array([spio.loadmat(self.info_path + '/coin_win' + version + '.mat',
                                                squeeze_me=True)['coin_win'][:n_runs]
                                   for version in reward_versions], dtype=bool).T

    def prepare_trial(self):

        # Switch sides if necessary
        current_run_lengths = np.array([self.run_lengths[self.i_episode[subj], subj] for subj in range(self.n_subj)])
        self.switched = self.n_rewards == current_run_lengths
        self.correct_box[self.switched] = 1 - self.correct_box[self.switched]

        # Keep track of number of rewards obtained in block, and block index
        self.n_rewards[self.switched] = 0
        self.i_episode[self.switched] += 1

    def produce_reward(self, action):

        # Check if action was correct
        correct_choice = action == self.correct_box

        # Look up the rewards that are scheduled for the current trial
        scheduled_coin_wins = np.array([self.coin_wins[self.n_correct[subj], subj] for subj in range(self.n_subj)])

        # Check for which subjects rewards need to be exchanged
        needs_reward_switch = np.array(self.switched * np.invert(scheduled_coin_wins))
        for subj in self.subj_ids:
            if needs_reward_switch[subj]:
                subj_scheduled_rewards = np.argwhere(self.coin_wins[:, subj])
                next_scheduled_reward = min(subj_scheduled_rewards[subj_scheduled_rewards > self.n_correct[subj]])
                self.coin_wins[self.n_correct[subj], subj] = True  # add reward to current index
                self.coin_wins[next_scheduled_reward, subj] = False  # remove reward from next index

        # Recalculate scheduled rewards based on updated coin_wins
        scheduled_coin_wins = np.array([self.coin_wins[self.n_correct[subj], subj] for subj in range(self.n_subj)])
        scheduled_coin_wins[np.invert(correct_choice)] = False  # no coins for incorrect actions

        # Keep track of things
        self.n_rewards[scheduled_coin_wins] += 1
        self.n_correct[action == self.correct_box] += 1

        return scheduled_coin_wins.astype(int)
