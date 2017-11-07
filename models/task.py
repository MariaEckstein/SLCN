import numpy as np
import scipy.io as spio


class Task(object):
    def __init__(self, task_stuff):
        self.n_actions = task_stuff['n_actions']
        self.reward_prob = task_stuff['reward_prob']
        self.specifications = task_stuff['specifications']
        self.run_length = spio.loadmat(self.specifications + '/run_length0.mat', squeeze_me=True)['run_length']
        self.coin_win = spio.loadmat(self.specifications + '/coin_win0.mat', squeeze_me=True)['coin_win']
        self.correct_box = np.random.rand() > 0.5
        self.n_rewards = 0
        self.n_correct = 0
        self.i_episode = 0

    def produce_reward(self, action):
        correct_choice = action == self.correct_box
        if correct_choice:
            reward = self.coin_win[self.n_correct]
            self.n_correct += 1
            self.n_rewards += reward
        else:
            reward = 0
        return reward

    def switch_box(self):
        switch = 0
        period_over = self.n_rewards == self.run_length[self.i_episode]
        if period_over:
            self.correct_box = 1 - self.correct_box
            self.n_rewards = 0
            self.i_episode += 1
            switch = 1
        return switch

