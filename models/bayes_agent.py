import numpy as np

# Problem: ACC is probably not coded the same way as in humans; is switch trial coded the same way?


class BayesAgent(object):
    def __init__(self, agent_stuff, task_stuff):
        self.name = agent_stuff['name']
        self.id = agent_stuff['id']
        self.initial_switch_prob = agent_stuff['initial_switch_prob']
        self.switch_prob = self.initial_switch_prob
        self.reward_prob = task_stuff['reward_prob']
        self.impatience = agent_stuff['impatience']
        self.n_actions = task_stuff['n_actions']
        self.p_boxes = np.ones(self.n_actions) / self.n_actions
        self.previous_action = np.nan
        self.just_switched = False

    def take_action(self):
        action = np.argwhere(self.p_boxes == np.max(self.p_boxes))[0]
        self.just_switched = not action == self.previous_action
        self.previous_action = action
        return action

    def learn(self, action, reward):
        if reward == 1:  # The probability of getting a reward is 0 for all actions except the right one
            lik_boxes = np.zeros(self.n_actions) + 0.001  # avoid getting likelihoods of 0
            lik_boxes[action] = self.reward_prob
        else:  # The probability of getting NO reward is 1 for all actions except the right one
            lik_boxes = np.ones(self.n_actions)
            lik_boxes[action] = 1 - self.reward_prob

        lik_times_prior = lik_boxes * self.p_boxes  # * = element-wise multiplication
        posterior = lik_times_prior / np.sum(lik_times_prior)  # normalize such that the sum == 1

        # if self.just_switched and reward == 1:
        #     self.switch_prob = 0
        # else:
        #     self.switch_prob += (0.5 - self.switch_prob) * self.impatience

        self.p_boxes = posterior * (1 - self.switch_prob) + np.flipud(posterior) * self.switch_prob
