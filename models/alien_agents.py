import numpy as np


class Agent(object):
    def __init__(self, agent_stuff, all_params_lim, task_stuff=np.nan):

        # Get parameters
        self.n_actions = agent_stuff['n_actions']
        self.n_TS = agent_stuff['n_TS']
        self.method = agent_stuff['method']
        self.learning_style = agent_stuff['learning_style']
        self.id = agent_stuff['id']
        [self.alpha, self.beta, self.epsilon, self.perseverance, self.decay, self.mix] = all_params_lim
        self.alpha_high = self.alpha  # TD
        self.decay_high = self.decay  # TD

        # Set up values at high (context-TS) level and at low level (stimulus-action)
        self.initial_q = 1 / self.n_actions
        self.Q_low = self.initial_q * np.ones([self.n_TS, 4, self.n_actions]) +\
            np.random.normal(0, 0.01, [self.n_TS, 4, self.n_actions])  # jitter to avoid identical values
        self.Q_high = self.initial_q * np.ones(self.n_actions) + np.random.normal(0, 0.01, self.n_actions)

        # Initialize action probs, current TS, previous_action, LL
        self.p_actions = np.ones(self.n_actions) / self.n_actions
        self.p_TS = np.ones(self.n_TS) / self.n_TS
        self.TS = np.nan
        self.previous_action = []
        self.LL = 0

    def select_action(self, stimulus):
        self.select_TS(stimulus[0])
        self.calculate_p_actions(stimulus)
        action = np.random.choice(range(self.n_actions), p=self.p_actions)
        self.previous_action = action
        return action

    def learn(self, stimulus, action, reward):
        self.decay_values()
        self.update_Qs(stimulus, action, reward)
        self.update_LL(action)

    def calculate_p_actions(self, stimulus):
        sticky_Q_low = self.make_sticky(self.Q_low[self.TS, stimulus[1], :])
        if self.method == 'epsilon-greedy':
            self.calculate_p_epsilon_greedy(sticky_Q_low)
        elif self.method == 'softmax':
            self.calculate_p_softmax(sticky_Q_low)

    def select_TS(self, context):
        if self.learning_style == 'hierarchical':
            # Could add stickiness, decay, epsilon, some kind of self.p_TS?
            self.TS = np.argmax(self.Q_high)  # ?!? what about rapid changes? TS switches?
        else:
            self.TS = context

    def make_sticky(self, Q):
        Q[self.previous_action] += self.perseverance
        # Q[Q < 0.001] = 0.001
        # Q[Q > 0.999] = 0.999
        return Q

    def calculate_p_epsilon_greedy(self, Q_low):
        self.p_actions = self.epsilon / (self.n_actions - 1) * np.ones(self.n_actions)  # all other act. share epsilon
        self.p_actions[np.argmax(Q_low)] = 1 - self.epsilon

    def calculate_p_softmax(self, Q_low):
        exp_2_0 = np.exp(self.beta * (Q_low[2] - Q_low[0]))
        exp_1_0 = np.exp(self.beta * (Q_low[1] - Q_low[0]))
        exp_2_1 = np.exp(self.beta * (Q_low[2] - Q_low[1]))
        exp_0_1 = np.exp(self.beta * (Q_low[0] - Q_low[1]))
        self.p_actions[0] = 1 / (1 + exp_2_0 + exp_1_0)
        self.p_actions[1] = 1 / (1 + exp_2_1 + exp_0_1)
        self.p_actions[2] = 1 - self.p_actions[0] - self.p_actions[1]

    def decay_values(self):
        self.Q_low += self.decay * (self.initial_q - self.Q_low)
        self.Q_high += self.decay_high * (self.initial_q - self.Q_high)

    def update_Qs(self, stimulus, action, reward):
        # Update Q_low
        RPE_low = reward - self.Q_low[self.TS, stimulus[1], action]
        self.Q_low[self.TS, stimulus[1], action] += self.alpha * RPE_low
        # Update Q_high
        RPE_high = reward - self.Q_high[action]
        RPE_mix = self.mix * RPE_high + (1 - self.mix) * RPE_low
        self.Q_high[self.TS] += self.alpha_high * RPE_mix

    def update_LL(self, action):
        self.LL += np.log(self.p_actions[action])
