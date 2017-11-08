import numpy as np

# TDs:
# - model fitting


class RLAgent(object):
        def __init__(self, agent_stuff, task, goal):
            self.goal = goal
            # Task features
            self.n_actions = task.n_actions
            self.initial_value = 1 / self.n_actions
            # Agent features
            self.name = agent_stuff['name']
            self.id = agent_stuff['id']
            self.alpha = agent_stuff['alpha']  # learning rate
            self.epsilon = agent_stuff['epsilon']  # greediness
            self.beta = agent_stuff['beta']  # softmax temperature
            self.perseverance = agent_stuff['perseverance']  # sticky choice
            self.decay = agent_stuff['decay']  # how fast do values decay back to uniform?
            self.method = agent_stuff['method']  # epsilon-greedy or softmax?
            self.q = self.initial_value * np.ones(self.n_actions)
            self.previous_action = np.random.choice(range(self.n_actions))

        def take_action(self):
            action = self.select_action()
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

        def learn(self, action, reward):
            self.q += self.decay * (1 / self.n_actions - self.q)  # decay values back to uniform
            self.q[action] += self.alpha * (reward - self.q[action])  # update value of chosen action

        def __is_greedy(self):
            return np.random.rand() > self.epsilon
