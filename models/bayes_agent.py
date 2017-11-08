import numpy as np
import pandas as pd


class BayesAgent(object):
    def __init__(self, agent_stuff, task, goal):
        self.goal = goal
        # Agent features
        self.name = agent_stuff['name']
        self.id = agent_stuff['id']
        self.file_name = agent_stuff['data_path'] + '/PS_' + str(self.id) + '.csv'
        if self.goal == 'model_data':
            self.actions = pd.read_csv(self.file_name)['selected_box']
        # Task features
        self.p_switch = 1 / np.mean(task.run_length)  # true average switch probability
        self.p_reward = task.p_reward  # true reward probability
        self.n_actions = task.n_actions  # 2
        # Probability of each action (box) to be the "magic" box, i.e., the one that currently gives rewards
        self.p_boxes = np.ones(self.n_actions) / self.n_actions  # initialize prior uniformly over all actions
        # Estimating p_switch and p_reward from data
        # TD

    def take_action(self, trial):
        if self.goal == 'model_data':
            action = int(self.actions[trial])
        else:
            action = int(np.random.rand() > self.p_boxes[0])  # select left ('0') when np.random.rand() < p_left
        return action

    def learn(self, action, reward):
        if reward == 1:  # The probability of getting a reward is 0 for all actions except the correct one
            lik_boxes = np.zeros(self.n_actions) + 0.001  # avoid getting likelihoods of 0
            lik_boxes[action] = self.p_reward
        else:  # The probability of getting NO reward is 1 for all actions except the correct one
            lik_boxes = np.ones(self.n_actions)
            lik_boxes[action] = 1 - self.p_reward

        lik_times_prior = lik_boxes * self.p_boxes  # * = element-wise multiplication
        posterior = lik_times_prior / np.sum(lik_times_prior)  # normalize such that sum == 1

        self.p_boxes = posterior * (1 - self.p_switch) + np.flipud(posterior) * self.p_switch
