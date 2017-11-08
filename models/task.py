import numpy as np
import scipy.io as spio
import pandas as pd


class Task(object):
    def __init__(self, task_stuff, agent_stuff, goal, ag, n_trials):
        self.goal = goal
        self.n_actions = task_stuff['n_actions']
        self.p_reward = task_stuff['p_reward']
        self.path = task_stuff['path']
        if self.goal == 'model_data':
            self.file_name = agent_stuff['data_path'] + '/PS_' + str(ag) + '.csv'
            agent_data = pd.read_csv(self.file_name)
            self.rewards = agent_data['reward']
            self.switch_trial = agent_data['switch_trial']
        self.n_trials = n_trials if goal == 'produce_data' else max(agent_data['TrialID'])
        self.reward_version = str(ag % 4)
        self.run_length = spio.loadmat(self.path + '/run_length' + self.reward_version + '.mat', squeeze_me=True)['run_length']
        self.coin_win = spio.loadmat(self.path + '/coin_win' + self.reward_version + '.mat', squeeze_me=True)['coin_win']
        self.correct_box = np.random.rand() > 0.5
        self.n_rewards = 0
        self.n_correct = 0
        self.i_episode = 0
        self.switched = False

    def produce_reward(self, action, trial):
        if self.goal == 'model_data':
            reward = int(self.rewards[trial])
        else:
            correct_choice = action == self.correct_box
            if correct_choice:
                reward = self.coin_win[self.n_correct]  # get predetermined reward
                if reward == 0 and self.switched:  # the first trial after a switch is always rewarded
                    reward = 1
                    self.switch_rewards()
                self.n_correct += 1
                self.n_rewards += reward
            else:
                reward = 0
        return reward

    def switch_rewards(self):
        reward_indexes = np.argwhere(self.coin_win == 1)
        next_reward_index = reward_indexes[reward_indexes > self.n_correct][0]
        self.coin_win[next_reward_index] = 0
        self.switched = False

    def switch_box(self, trial):
        if self.goal == 'model_data':
            period_over = self.switch_trial[trial]
        else:
            period_over = self.n_rewards == self.run_length[self.i_episode]
        if period_over:
            self.correct_box = 1 - self.correct_box
            self.switched = True
            self.n_rewards = 0
            self.i_episode += 1

