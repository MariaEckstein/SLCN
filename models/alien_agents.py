import numpy as np
import math


class Agent(object):
    def __init__(self, agent_stuff, all_params_lim, task_stuff=np.nan):

        # Get parameters
        self.n_actions = task_stuff['n_actions']
        self.n_TS = agent_stuff['n_TS']
        self.method = agent_stuff['method']
        self.learning_style = agent_stuff['learning_style']
        self.select_deterministic = self.learning_style == 'flat'  # Hack to select the right TS each time for the flat agent
        self.mix_probs = agent_stuff['mix_probs']
        self.id = agent_stuff['id']
        [self.alpha, self.beta, self.epsilon] = all_params_lim
        self.alpha_high = self.alpha  # TD
        assert(self.alpha > 0)  # Make sure that alpha is a number and is > 0
        assert(self.mix_probs in [True, False])
        assert(self.method in ['epsilon-greedy', 'softmax'])
        assert(self.learning_style in ['flat', 'hierarchical'])

        # Set up values at low (stimulus-action) level
        self.initial_q_low = 5 / 3  # 3 possible actions, 5 was the reward during training
        if self.learning_style == 'flat':
            n_TS = task_stuff['n_contexts']
        elif self.learning_style == 'hierarchical':
            n_TS = self.n_TS
        Q_low_dim = [n_TS, task_stuff['n_aliens'], self.n_actions]
        self.Q_low = self.initial_q_low * np.ones(Q_low_dim) + np.random.normal(0, 0.001, Q_low_dim)  # jitter avoids ident. values

        # Set up values at high (context-TS) level
        if self.learning_style == 'flat':
            self.Q_high = np.eye(task_stuff['n_contexts'])  # agent always selects the appropriate table
        elif self.learning_style == 'hierarchical':
            self.initial_q_high = 5 / 3  # 5 was the reward during training
            Q_high_dim = [task_stuff['n_contexts'], self.n_TS]
            self.Q_high = self.initial_q_high * np.ones(Q_high_dim) + np.random.normal(0, 0.001, Q_high_dim)

        # Initialize action probs, current TS and action, LL
        self.p_TS = np.ones(self.n_TS) / self.n_TS  # P(TS|context)
        self.p_actions = np.ones(self.n_actions) / self.n_actions  # P(action|TS)
        self.TS = []
        self.prev_action = []
        self.LL = 0

    def select_action(self, stimulus):

        self.p_TS = self.get_p_from_Q(Q=self.Q_high[stimulus[0], :],
                                      select_deterministic=self.select_deterministic)  # works for flat & hierarchical
        p_actions = [self.get_p_from_Q(Q=self.Q_low[TS_i, stimulus[1], :])
                     for TS_i in range(self.n_TS)]

        if self.mix_probs:  # P(action|context, alien) = \sum_TS P(action|TS_i, alien) P(TS_i|context)
            self.TS = np.argmax(self.p_TS)  # just for saving in agent_data
            self.p_actions = np.sum(np.dot(self.p_TS * np.eye(self.n_TS), p_actions), axis=0)  # * = elementwise multiplication
        else:  # Select one TS based on Q(TS|context), then select one action based on Q(action|context, alien)
            self.TS = np.random.choice(range(self.n_TS), p=self.p_TS)
            self.p_actions = p_actions[self.TS]

        self.prev_action = np.random.choice(range(self.n_actions), p=self.p_actions)
        return self.prev_action

    def learn(self, stimulus, action, reward):
        [old_Q_a, RPE_a, new_Q_a, old_Q_TS, RPE_TS, new_Q_TS] = self.update_Qs(stimulus, action, reward)
        self.update_LL(action)
        return [old_Q_a, RPE_a, new_Q_a, old_Q_TS, RPE_TS, new_Q_TS]

    def get_p_from_Q(self, Q, select_deterministic=False):
        if select_deterministic:
            assert(np.sum(Q == 1) == 1)  # Verify that exactly 1 Q-value == 1
            p = np.array(Q == 1, dtype=int)  # select the one TS that has a value of 1 (all others have values of 0)
        elif self.method == 'epsilon-greedy':
            p = self.calculate_p_epsilon_greedy(Q)
        elif self.method == 'softmax':
            p = self.calculate_p_softmax(Q)
        assert math.isclose(sum(p), 1, rel_tol=1e-5)  # Verify that probabilities sum up to 1
        return p

    def calculate_p_epsilon_greedy(self, Q):
        p_actions = self.epsilon / (self.n_actions - 1) * np.ones(self.n_actions)  # all other act. share epsilon
        p_actions[np.argmax(Q)] = 1 - self.epsilon  # the action with the largest Q value gets all the rest
        return p_actions

    def calculate_p_softmax(self, Q):
        p_actions = np.empty(len(Q))
        for i in range(len(Q)):
            denominator = 1 + sum([np.exp(self.beta * (Q[j] - Q[i])) for j in range(len(Q)) if j != i])
            p_actions[i] = 1 / denominator
        return p_actions

    def update_Qs(self, stimulus, action, reward):
        old_Q_low = self.Q_low[self.TS, stimulus[1], action].copy()
        old_Q_high = self.Q_high[stimulus[0], self.TS].copy()
        RPEs_low = reward - self.Q_low[:, stimulus[1], action]  # Q_low for all TSs, given alien & action
        RPEs_high = reward - self.Q_high[stimulus[0], :]  # Q_high for all TSs, given context
        if self.mix_probs:
            self.Q_low[:, stimulus[1], action] += self.alpha * self.p_TS * RPEs_low  # flat agent: p_TS has just one 1
            if self.learning_style == 'hierarchical':
                self.Q_high[stimulus[0], :] += self.alpha_high * self.p_TS * RPEs_high
        else:
            self.Q_low[self.TS, stimulus[1], action] += self.alpha * RPEs_low[self.TS]
            if self.learning_style == 'hierarchical':
                self.Q_high[stimulus[0], self.TS] += self.alpha_high * RPEs_high[self.TS]
        # Return old values, RPE, new values
        new_Q_high = self.Q_high[stimulus[0], self.TS].copy()
        new_Q_low = self.Q_low[self.TS, stimulus[1], action].copy()
        return [old_Q_low, RPEs_low[self.TS], new_Q_low, old_Q_high, RPEs_high[self.TS], new_Q_high]

    def update_LL(self, action):
        self.LL += np.log(self.p_actions[action])
        # assert action == self.prev_action  # obviously fails when called in calculate_NLL
