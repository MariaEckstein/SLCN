import numpy as np

# TDs:
# - softmax instead epsilon greedy

class RLAgent(object):
        def __init__(self, agent_stuff):
            self.id = agent_stuff['id']
            # Agent's RL features
            self.initial_value = 0.5
            self.alpha = agent_stuff['alpha']  # learning rate
            self.epsilon = agent_stuff['epsilon']  # greediness
            self.beta = agent_stuff['beta']  # softmax temperature
            # Agent's values
            self.q = self.initial_value * np.ones(2)  # curiosity about basic actions and already-discovered options

        # Take_action and helpers
        def take_action(self):
            return self.__select_action()

        def __select_action(self):
            if self.__is_greedy():
                selected_actions = np.argwhere(self.q == np.nanmax(self.q))  # all actions with the highest value
            else:
                selected_actions = np.argwhere(~np.isnan(self.q))
            select = np.random.randint(len(selected_actions))  # randomly select the index of one of the actions
            return selected_actions[select]  # pick that action

        # Learn and helpers
        def learn(self, action, reward):
            self.q[action] += self.alpha * (reward - self.q[action])

        # Little helpers
        def __is_greedy(self):
            return np.random.rand() > self.epsilon
