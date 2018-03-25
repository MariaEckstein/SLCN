import numpy as np
import math


class Agent(object):
    def __init__(self, agent_stuff, all_params_lim, task_stuff=np.nan):

        # Get parameters
        self.n_actions = task_stuff['n_actions']
        self.n_TS = agent_stuff['n_TS']
        self.method = agent_stuff['method']
        self.learning_style = agent_stuff['learning_style']
        self.id = agent_stuff['id']
        [self.alpha, self.beta, self.epsilon, self.perseverance, self.forget, self.mix] = all_params_lim
        self.alpha_high = self.alpha  # TD
        self.forget_high = self.forget  # TD

        # Set up values at low (stimulus-action) level
        self.initial_q_low = 5 / 3  # 3 possible actions, 5 was the reward during training
        if self.learning_style == "flat":
            first_dim = task_stuff['n_contexts']
        else:
            first_dim = self.n_TS
        Q_low_dim = [first_dim, task_stuff['n_aliens'], self.n_actions]
        self.Q_low = self.initial_q_low * np.ones(Q_low_dim) + np.random.normal(0, 0.001, Q_low_dim)  # jitter avoids ident. values

        # Set up values at high (context-TS) level
        self.initial_q_high = 5 / 3  # 5 was the reward during training
        Q_high_dim = [task_stuff['n_contexts'], self.n_TS]
        self.Q_high = self.initial_q_high * np.ones(Q_high_dim) + np.random.normal(0, 0.001, Q_high_dim)

        # Initialize action probs, current TS and action, LL
        self.p_TS = np.ones(self.n_TS) / self.n_TS
        self.p_actions = np.ones(self.n_actions) / self.n_actions
        self.TS = []
        self.action = []
        self.LL = 0

    def select_action(self, stimulus):
        # Step 1: select TS
        if self.learning_style == 'flat':
            self.p_TS = np.full(self.n_TS, np.nan)  # should be n_contexts
            self.TS = stimulus[0]  # Current context
        else:
            self.p_TS = self.get_p_from_Q(self.Q_high[stimulus[0], :], self.TS)
            self.TS = np.random.choice(range(self.n_TS), p=self.p_TS)

        # Step 2: select action based on TS
        self.p_actions = self.get_p_from_Q(self.Q_low[self.TS, stimulus[1], :], self.action)
        self.action = np.random.choice(range(self.n_actions), p=self.p_actions)
        return self.action

    def learn(self, stimulus, action, reward):
        self.forget_Qs()
        self.update_Qs(stimulus, action, reward)
        self.update_LL(action)

    def get_p_from_Q(self, Q, previous_choice):
        Q[previous_choice] += self.perseverance  # Should affect the original Q table! -> Perseverance is persistent!
        if self.method == 'epsilon-greedy':
            p = self.calculate_p_epsilon_greedy(Q)
        else:  # if self.method == 'softmax':
            p = self.calculate_p_softmax(Q)
        return p

    def calculate_p_epsilon_greedy(self, Q):
        p_actions = self.epsilon / (self.n_actions - 1) * np.ones(self.n_actions)  # all other act. share epsilon
        p_actions[np.argmax(Q)] = 1 - self.epsilon
        assert math.isclose(sum(p_actions), 1, rel_tol=1e-5)
        return p_actions

    def calculate_p_softmax(self, Q):
        p_actions = np.empty(len(Q))
        for i in range(len(Q)):
            denominator = 1 + sum([np.exp(self.beta * (Q[j] - Q[i])) for j in range(len(Q)) if j != i])
            p_actions[i] = 1 / denominator
        assert math.isclose(sum(p_actions), 1, rel_tol=1e-5)
        return p_actions

    def forget_Qs(self):
        self.Q_low += self.forget * (self.initial_q_low - self.Q_low)
        self.Q_high += self.forget_high * (self.initial_q_high - self.Q_high)

    def update_Qs(self, stimulus, action, reward):
        # Update Q_low
        RPE_low = reward - self.Q_low[self.TS, stimulus[1], action]
        self.Q_low[self.TS, stimulus[1], action] += self.alpha * RPE_low
        # Update Q_high
        RPE_high = reward - self.Q_high[stimulus[0], self.TS]
        RPE_mix = self.mix * RPE_high + (1 - self.mix) * RPE_low
        self.Q_high[stimulus[0], self.TS] += self.alpha_high * RPE_mix

    def update_LL(self, action):
        self.LL += np.log(self.p_actions[action])
        # assert action == self.action  # obviously fails when called in calculate_NLL
