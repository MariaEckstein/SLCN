import numpy as np

# TDs:
# - model fitting


class RLAgent(object):
        def __init__(self, agent_stuff, task_stuff):
            self.name = agent_stuff['name']
            self.id = agent_stuff['id']
            self.n_actions = task_stuff['n_actions']
            self.initial_value = 1 / self.n_actions
            self.alpha = agent_stuff['alpha']  # learning rate
            self.epsilon = agent_stuff['epsilon']  # greediness
            self.beta = agent_stuff['beta']  # softmax temperature
            self.perseverance = agent_stuff['perseverance']
            self.method = agent_stuff['method']
            self.q = self.initial_value * np.ones(self.n_actions)  # curiosity about basic actions and already-discovered options
            self.previous_action = np.random.choice(range(self.n_actions))
            self.just_switched = False

        def take_action(self):
            action = self.select_action()
            self.just_switched = not action == self.previous_action
            self.previous_action = action
            return action

        def select_action(self):
            if self.method == 'epsilon-greedy':
                if self.__is_greedy():
                    selected_actions = np.argwhere(self.q == np.nanmax(self.q))  # all actions with the highest value
                else:
                    selected_actions = np.argwhere(~np.isnan(self.q))
                select = np.random.randint(len(selected_actions))  # randomly select the index of one of the actions
                action = selected_actions[select]  # pick that action
            elif self.method == 'softmax':
                sticky_choice = np.zeros(self.n_actions)
                sticky_choice[self.previous_action] = 1
                p_left = 1 / (1 + np.exp(
                    self.beta * (self.q[1] - self.q[0]) +
                    self.perseverance * (sticky_choice[1] - sticky_choice[0])
                ))
                p_left = (1 - 0.001) * p_left + 0.001 / 2
                if np.random.rand() < p_left:
                    action = 0
                else:
                    action = 1
            return action

        def learn(self, action, reward):
            self.q[action] += self.alpha * (reward - self.q[action])

        def __is_greedy(self):
            return np.random.rand() > self.epsilon
