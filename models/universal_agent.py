import numpy as np
import pandas as pd


class UniversalAgent(object):
    def __init__(self, agent_stuff, task, goal):
        self.goal = goal
        # Agent features
        self.name = agent_stuff['name']
        self.id = agent_stuff['id']
        if self.goal == 'model_data':
            self.LL = 0
            self.file_name = agent_stuff['data_path'] + '/PS_' + str(self.id) + '.csv'
            self.actions = pd.read_csv(self.file_name)['selected_box']
        # Task features
        self.p_switch = 1 / np.mean(task.run_length)  # true average switch probability
        self.p_reward = task.p_reward  # true reward probability
        self.n_actions = task.n_actions  # 2
        # RL variables
        self.initial_value = 1 / self.n_actions
        self.alpha = agent_stuff['alpha']  # learning rate
        self.epsilon = agent_stuff['epsilon']  # greediness
        self.beta = agent_stuff['beta']  # softmax temperature
        self.perseverance = agent_stuff['perseverance']  # sticky choice
        self.decay = agent_stuff['decay']  # how fast do values decay back to uniform?
        self.method = agent_stuff['method']  # epsilon-greedy or softmax?
        self.q = self.initial_value * np.ones(self.n_actions)
        self.previous_action = np.random.choice(range(self.n_actions))
        # Bayes variables
        # Probability of each action (box) to be the "magic" box, i.e., the one that currently gives rewards
        self.p_boxes = np.ones(self.n_actions) / self.n_actions  # initialize prior uniformly over all actions
        # Estimating p_switch and p_reward from data
        # TD

    # Take action
    def take_action(self, trial):
        if self.goal == 'model_data':
            action = int(self.actions[trial])
        else:
            action = self.__select_action()
        self.__get_LL(action)
        return action

    def __get_LL(self, action):
        if self.name == 'RL':
            self.q = (1 - 0.001) * self.q + 0.001 / 2  # avoid 0's
            self.LL += np.log(self.q[action])
        else:
            self.LL += np.log(self.p_boxes[action])

    def __select_action(self):
        if self.name == 'RL':
            return self.__select_action_rl()
        else:
            return self.__select_action_bayes()

    def __select_action_bayes(self):
        return int(np.random.rand() > self.p_boxes[0])  # select left ('0') when np.random.rand() < p_left

    def __select_action_rl(self):
        if self.method == 'epsilon-greedy':
            if self.__is_greedy():
                selected_actions = np.argwhere(self.q == np.nanmax(self.q))  # all actions with the highest value
            else:
                selected_actions = np.argwhere(~np.isnan(self.q))
            select = np.random.randint(len(selected_actions))  # randomly select the index of one of the actions
            action = selected_actions[select]  # pick that action
        else:  # self.method == 'softmax':
            sticky_choice = np.zeros(self.n_actions)
            sticky_choice[self.previous_action] = 1
            p_left = 1 / (1 + np.exp(
                self.beta * (self.q[1] - self.q[0]) +
                self.perseverance * (sticky_choice[1] - sticky_choice[0])
            ))
            p_left = (1 - 0.001) * p_left + 0.001 / 2  # avoid 0's
            action = int(np.random.rand() > p_left)  # select left ('0') when np.random.rand() < p_left
        return action

    # Learn
    def learn(self, action, reward):
        if self.name == 'RL':
            self.__learn_rl(action, reward)
        else:
            self.__learn_bayes(action, reward)

    def __learn_rl(self, action, reward):
        self.q += self.decay * (1 / self.n_actions - self.q)  # decay values back to uniform
        self.q[action] += self.alpha * (reward - self.q[action])  # update value of chosen action

    def __learn_bayes(self, action, reward):
        if reward == 1:  # The probability of getting a reward is 0 for all actions except the correct one
            lik_boxes = np.zeros(self.n_actions) + 0.001  # avoid getting likelihoods of 0
            lik_boxes[action] = self.p_reward
        else:  # The probability of getting NO reward is 1 for all actions except the correct one
            lik_boxes = np.ones(self.n_actions)
            lik_boxes[action] = 1 - self.p_reward
        lik_times_prior = lik_boxes * self.p_boxes  # * = element-wise multiplication
        posterior = lik_times_prior / np.sum(lik_times_prior)  # normalize such that sum == 1
        self.p_boxes = posterior * (1 - self.p_switch) + np.flipud(posterior) * self.p_switch

    # Little helpers
    def __is_greedy(self):
        return np.random.rand() > self.epsilon
