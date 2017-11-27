import numpy as np
import pandas as pd


class UniversalAgent(object):
    def __init__(self, model, goal, params, task, id):
        self.n_actions = task.n_actions
        self.learning_style = model.agent_stuff['learning_style']
        self.id = id
        self.method = model.agent_stuff['method']
        raw_pars = model.parameters.get_pars(model.agent_stuff, params)
        pars = model.parameters.constrain_limits(model.parameters.sigmoid(raw_pars))  # only simulate in reasonable range
        [self.alpha, self.beta, self.epsilon, self.perseverance, self.decay] = pars
        # Load participant data
        self.data_path = model.agent_stuff['data_path']
        self.hist_path = model.agent_stuff['hist_path']
        if goal == 'simulate':
            self.RTs = np.nan
        else:  # goal == 'calculate_fit' or 'calculate_NLL'
            file_name = self.data_path + '/PS_' + str(self.id) + '.csv'
            agent_data = pd.read_csv(file_name)
            self.actions = agent_data['selected_box']
            self.RTs = agent_data['RT']
        # Keep track of things
        if self.learning_style == 'RL':
            self.initial_value = 1 / self.n_actions
            self.q = self.initial_value * np.ones(self.n_actions)
        elif self.learning_style == 'Bayes':
            self.p_switch = 1 / np.mean(task.run_length / task.p_reward)  # true average switch probability
            self.p_reward = task.p_reward  # true reward probability
            self.p_boxes = np.ones(self.n_actions) / self.n_actions  # initialize prior uniformly over all actions
            self.initial_value = self.p_boxes
        self.previous_action = np.zeros(self.n_actions)
        self.LL = 0
        self.p_actions = 0.5 * np.ones(self.n_actions)

    # Take action
    def take_action(self, trial, goal):
        self._calculate_p_actions(self.method)
        action = self._select_action(goal, trial)
        self._update_LL(action)
        return action

    # Get LL
    def _update_LL(self, action):
        self.p_actions = (1 - 0.001) * self.p_actions + 0.001 / self.n_actions  # avoid 0's
        self.LL += np.log(self.p_actions[action])

    # Action helpers
    def _calculate_p_actions(self, method):
        action_values = self._get_action_values()
        sticky_values = action_values + self.perseverance * self.previous_action
        sticky_values[sticky_values <= 0] = 0.001
        if method == 'epsilon-greedy':
            best_actions = np.argwhere(sticky_values == np.nanmax(sticky_values))  # actions with highest value
            n_best_actions = len(best_actions)
            n_other_actions = self.n_actions - n_best_actions
            epsilon = 0 if n_other_actions == 0 else self.epsilon / n_other_actions
            self.p_actions = epsilon * np.ones(self.n_actions)
            self.p_actions[best_actions] = (1 - self.epsilon) / n_best_actions
        elif method == 'softmax':
            p_left_box = 1 / (1 + np.exp(
                self.beta * (sticky_values[1] - sticky_values[0])
            ))
            self.p_actions = np.array([p_left_box, 1 - p_left_box])
        elif method == 'direct':
            self.p_actions = sticky_values / np.sum(sticky_values)  # normalize

    def _select_action(self, goal, trial):
        if goal == 'simulate':
            action = int(np.random.rand() > self.p_actions[0])  # select left ('0') when np.random.rand() < p_actions[0]
        else:
            action = int(self.actions[trial])  # look up which action participant actually took
        self.previous_action = np.array(range(self.n_actions)) == action
        return action

    def _get_action_values(self):
        if self.learning_style == 'RL':
            return self.q
        else:
            return self.p_boxes

    # Learn
    def learn(self, action, reward):
        if self.learning_style == 'RL':
            self._learn_rl(action, reward)
        elif self.learning_style == 'Bayes':
            self._learn_bayes(action, reward)

    # Learn helpers
    def _learn_bayes(self, action, reward):
        self.p_boxes += self.decay * (self.initial_value - self.p_boxes)  # decay values back to uniform
        if reward == 1:  # The probability of getting a reward is 0 for all actions except the correct one
            lik_boxes = np.zeros(self.n_actions) + 0.001  # avoid getting likelihoods of 0
            lik_boxes[action] = self.p_reward
        else:  # The probability of getting NO reward is 1 for all actions except the correct one
            lik_boxes = np.ones(self.n_actions)
            lik_boxes[action] = 1 - self.p_reward
        lik_times_prior = lik_boxes * self.p_boxes  # * = element-wise multiplication
        posterior = lik_times_prior / np.sum(lik_times_prior)  # normalize such that sum == 1
        self.p_boxes = posterior * (1 - self.p_switch) + np.flipud(posterior) * self.p_switch

    def _learn_rl(self, action, reward):
        self.q += self.decay * (self.initial_value - self.q)  # decay values back to initla value
        self.q[action] += self.alpha * (reward - self.q[action])  # update value of chosen action

    # Little helpers
    def _is_greedy(self):
        return np.random.rand() > self.epsilon
