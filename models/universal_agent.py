import numpy as np
import pandas as pd
import os


class UniversalAgent(object):
    def __init__(self, agent_stuff, params, task, id):
        self.n_actions = task.n_actions  # 2
        self.learning_style = agent_stuff['learning_style']
        self.id = id
        # Parameters
        self.free_par = agent_stuff['free_par']
        default_par = agent_stuff['default_par']
        self.alpha = params[0] if self.free_par[0] else default_par[0]  # learning rate
        self.beta = 30 * params[1] if self.free_par[1] else default_par[1]  # softmax temperature
        self.epsilon = params[2] if self.free_par[2] else default_par[2]  # greediness
        self.perseverance = 30 * params[3] if self.free_par[3] else default_par[3]  # sticky choice
        self.decay = params[4] if self.free_par[4] else default_par[4]  # how fast do values decay back to uniform?
        self.method = agent_stuff['method']  # epsilon-greedy, softmax, or direct?
        # Load participant data
        file_name = agent_stuff['data_path'] + '/PS_' + str(self.id) + '.csv'
        if os.path.isfile(file_name):
            self.actions = pd.read_csv(file_name)['selected_box']
        # Keep track of things
        if self.learning_style == 'RL':
            self.initial_value = 1 / self.n_actions
            self.q = self.initial_value * np.ones(self.n_actions)
        elif self.learning_style == 'Bayes':
            self.p_switch = 1 / np.mean(task.run_length)  # true average switch probability
            self.p_reward = task.p_reward  # true reward probability
            self.p_boxes = np.ones(self.n_actions) / self.n_actions  # initialize prior uniformly over all actions
        self.previous_action = np.zeros(self.n_actions)
        self.LL = 0

    # Take action
    def take_action(self, trial, goal):
        p_actions = self.__get_p_actions(self.method)
        action = self.__select_action(goal, p_actions, trial)
        self.__update_LL(action, p_actions)
        return action

    # Get LL
    def __update_LL(self, action, p_actions):
        p_actions = (1 - 0.001) * p_actions + 0.001 / self.n_actions  # avoid 0's
        self.LL += np.log(p_actions[action])

    # Action helpers
    def __get_p_actions(self, method):
        action_values = self.__get_action_values()
        if method == 'epsilon-greedy':
            sticky_values = action_values + self.perseverance * self.previous_action
            best_actions = np.argwhere(sticky_values == np.nanmax(sticky_values))  # actions with highest value
            n_best_actions = len(best_actions)
            n_other_actions = self.n_actions - n_best_actions
            epsilon = 0 if n_other_actions == 0 else self.epsilon / n_other_actions
            p_actions = epsilon * np.ones(self.n_actions)
            p_actions[best_actions] = (1 - self.epsilon) / n_best_actions
        elif method == 'softmax':
            p_left_box = 1 / (1 + np.exp(
                self.beta * (action_values[1] - action_values[0]) +
                self.perseverance * (self.previous_action[1] - self.previous_action[0])
            ))
            p_actions = np.array([p_left_box, 1 - p_left_box])
        elif method == 'direct':
            p_actions = action_values / np.sum(action_values)  # normalize
        return p_actions

    def __select_action(self, goal, p_actions, trial):
        if goal == 'simulate':
            action = int(np.random.rand() > p_actions[0])  # select left ('0') when np.random.rand() < p_actions[0]
        elif goal == 'model':
            action = int(self.actions[trial])  # look up which action participant actually took
        self.previous_action[:] = 0
        self.previous_action[action] = 1
        return action

    def __get_action_values(self):
        if self.learning_style == 'RL':
            return self.q
        else:
            return self.p_boxes

    # Learn
    def learn(self, action, reward):
        if self.learning_style == 'RL':
            self.__learn_rl(action, reward)
        elif self.learning_style == 'Bayes':
            self.__learn_bayes(action, reward)

    # Learn helpers
    def __learn_bayes(self, action, reward):
        self.p_boxes += self.decay * (1 / self.n_actions - self.p_boxes)  # decay values back to uniform
        if reward == 1:  # The probability of getting a reward is 0 for all actions except the correct one
            lik_boxes = np.zeros(self.n_actions) + 0.001  # avoid getting likelihoods of 0
            lik_boxes[action] = self.p_reward
        else:  # The probability of getting NO reward is 1 for all actions except the correct one
            lik_boxes = np.ones(self.n_actions)
            lik_boxes[action] = 1 - self.p_reward
        lik_times_prior = lik_boxes * self.p_boxes  # * = element-wise multiplication
        posterior = lik_times_prior / np.sum(lik_times_prior)  # normalize such that sum == 1
        self.p_boxes = posterior * (1 - self.p_switch) + np.flipud(posterior) * self.p_switch

    def __learn_rl(self, action, reward):
        self.q += self.decay * (1 / self.n_actions - self.q)  # decay values back to uniform
        self.q[action] += self.alpha * (reward - self.q[action])  # update value of chosen action

    # Little helpers
    def __is_greedy(self):
        return np.random.rand() > self.epsilon
