import numpy as np


class Task(object):
    def __init__(self, task_stuff):
        self.n_actions = task_stuff['n_actions']
        self.reward_prob = task_stuff['reward_prob']
        self.l_episodes = task_stuff['l_episodes']
        self.correct_box = np.random.rand() > 0.5
        self.n_rewards = 0
        self.i_episode = 0
        self.switched = False

    def produce_reward(self, action):
        correct_choice = action == self.correct_box
        lucky = np.random.rand() < self.reward_prob
        if correct_choice and (lucky or self.switched):
            reward = 1
            self.n_rewards += 1
            self.switched = False
        else:
            reward = 0
        return reward

    def switch_box(self):
        switch = 0
        period_over = self.n_rewards == self.l_episodes[self.i_episode]
        if period_over:
            self.correct_box = 1 - self.correct_box
            self.n_rewards = 0
            self.i_episode += 1
            self.switched = True
            switch = 1
        return switch

