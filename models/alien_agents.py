import numpy as np


class Agent(object):
    def __init__(self, agent_stuff, all_params_lim):
        self.initial_q = 1 / 3
        self.Q = self.initial_q * np.ones([3, 4, 3]) +\
                 (0.5 - np.random.rand(3, 4, 3)) / 1000  # jitter to avoid identical values
        self.Q_high = self.initial_q * np.ones(3) + (0.5 - np.random.rand(3)) / 1000
        self.previous_action = np.nan
        self.method = agent_stuff['method']
        [self.alpha, self.beta, self.epsilon, self.perseverance, self.decay,
         self.mix] = all_params_lim
        self.alpha_high = self.alpha
        self.decay_high = self.decay
        self.p_actions = np.ones(3) / 3
        self.TS = np.nan
        self.LL = 0

    def select_action(self, stimulus):
        self.calculate_p_actions(stimulus)
        action = np.random.choice(range(3), p=self.p_actions)
        self.previous_action = action
        return action

    def learn(self, stimulus, action, reward):
        self.decay_values()
        self.update_Qs(stimulus, action, reward)
        self.update_LL(action)

    def calculate_p_actions(self, stimulus):
        self.select_TS(stimulus[0])
        sticky_Q = self.make_sticky(self.Q[self.TS, stimulus[1], :])
        if self.method == 'epsilon-greedy':
            self.calculate_p_epsilon_greedy(sticky_Q)
        elif self.method == 'softmax':
            self.calculate_p_softmax(sticky_Q)

    def select_TS(self, context):
        if self.method == 'hierarchical':
            # all the fancy stuff is still missing: decay, perseverance, epsilon / softmax
            self.TS = np.argmax(self.Q_high)
        else:
            self.TS = context

    def make_sticky(self, Q):
        Q[self.previous_action] += self.perseverance
        Q[Q <= 0.001] = 0.001
        Q[Q >= 0.999] = 0.999
        return Q

    def calculate_p_epsilon_greedy(self, sticky_Q):
        largest_location = np.argmax(sticky_Q)
        self.p_actions = self.epsilon * np.ones(3)
        self.p_actions[largest_location] = 1 - self.epsilon

    def calculate_p_softmax(self, sticky_Q):
        exp_2_0 = np.exp(self.beta * (sticky_Q[2] - sticky_Q[0]))
        exp_1_0 = np.exp(self.beta * (sticky_Q[1] - sticky_Q[0]))
        exp_2_1 = np.exp(self.beta * (sticky_Q[2] - sticky_Q[1]))
        exp_0_1 = np.exp(self.beta * (sticky_Q[0] - sticky_Q[1]))
        self.p_actions[0] = 1 / (1 + exp_2_0 + exp_1_0)
        self.p_actions[1] = 1 / (1 + exp_2_1 + exp_0_1)
        self.p_actions[2] = 1 - self.p_actions[0] - self.p_actions[1]

    def decay_values(self):
        self.Q += self.decay * (self.initial_q - self.Q)
        self.Q_high += self.decay_high * (self.initial_q - self.Q_high)

    def update_Qs(self, stimulus, action, reward):
        RPE_low = reward - self.Q[self.TS, stimulus[1], action]
        RPE_high = reward - self.Q_high[action]
        RPE_mix = self.mix * RPE_high + (1 - self.mix) * RPE_low
        self.Q[self.TS, stimulus[1], action] += self.alpha * RPE_low
        self.Q_high[self.TS] += self.alpha_high * RPE_mix

    def update_LL(self, action):
        self.LL += np.log(self.p_actions[action])
