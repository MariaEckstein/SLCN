import numpy as np
import scipy.io as spio


class Task(object):
    def __init__(self, task_stuff, agent_id):
        self.n_actions = task_stuff['n_actions']
        self.p_reward = task_stuff['p_reward']
        self.n_trials = task_stuff['n_trials']
        self.path = task_stuff['path']
        self.correct_box = int(np.random.rand() > 0.5)
        self.n_rewards = 0
        self.n_correct = 0
        self.i_episode = 0
        self.switched = False
        self.reward_version = str(agent_id % 4)
        self.run_length = spio.loadmat(self.path + '/run_length' + self.reward_version + '.mat',
                                       squeeze_me=True)['run_length']
        self.coin_win = spio.loadmat(self.path + '/coin_win' + self.reward_version + '.mat',
                                     squeeze_me=True)['coin_win']

    def produce_reward(self, action):
        correct_choice = action == self.correct_box
        if correct_choice:
            reward = self.coin_win[self.n_correct]  # get predetermined reward
            if reward == 0 and self.switched:  # the first trial after a switch is always rewarded
                reward = 1
                self._exchange_rewards()
                self.switched = False
            self.n_correct += 1
            self.n_rewards += reward
        else:
            reward = 0
        return reward

    def _exchange_rewards(self):
        reward_indexes = np.argwhere(self.coin_win == 1)
        next_reward_index = reward_indexes[reward_indexes > self.n_correct][0]
        self.coin_win[next_reward_index] = 0

    def prepare_trial(self, trial=np.nan):
        # switch box if necessary
        period_over = self.n_rewards == self.run_length[self.i_episode]
        if period_over:
            self.correct_box = 1 - self.correct_box
            self.switched = True
            self.n_rewards = 0
            self.i_episode += 1

    @staticmethod
    def present_stimulus(context=(), trial=()):
        # not necessary here, but in other tasks
        return None
