import numpy as np


class RLAgent(object):

    def __init__(self, agent_stuff, all_pars, task_stuff=np.nan):

        # Get info about task
        self.n_actions, self.n_TS = task_stuff['n_actions'], agent_stuff['n_TS']

        # Get RL parameters
        [self.alpha, self.alpha_high,
         self.beta, self.beta_high,
         self.epsilon, self.forget, self.TS_bias] = all_pars
        self.forget_high = self.forget
        self.adjust_parameters(agent_stuff)

        # Set up value tables for TS values (Q_high) and action values (Q_low) for each TS
        self.Q_high = np.zeros(self.n_TS)
        self.Q_low = np.zeros([self.n_TS, self.n_actions])
        self.initial_Q = 1. / self.n_actions

        # Initialize RL features and log likelihood (LL)
        self.p_TS = np.ones(self.n_TS) / self.n_TS  # P(TS|context)
        self.p_actions = np.ones(self.n_actions) / self.n_actions  # P(action|TS)
        self.Q_actions = np.nan
        self.RPEs_low = np.nan
        self.RPEs_high = np.nan
        self.LL = 0

        # Initialize agent's "memory"
        self.action = []

    def adjust_parameters(self, agent_stuff):

        # Get betas and TS_bias on the right scale
        self.beta *= agent_stuff['beta_scaler']
        self.beta_high *= agent_stuff['beta_high_scaler']
        self.TS_bias *= agent_stuff['TS_bias_scaler']

        # Get the right high-level alpha and beta
        if self.alpha_high < 1e-5:
            self.alpha_high = self.alpha
        if self.beta_high < 1e-5:
            self.beta_high = self.beta

    def select_action(self, stimulus):

        # Calculate probabilities for each TS from Q_high, and select action according to associated Q_low
        self.p_TS = self.get_p_from_Q(Q=self.Q_high, beta=self.beta_high, epsilon=0)
        self.Q_actions = np.dot(self.p_TS, self.Q_low)  # Weighted average
        self.p_actions = self.get_p_from_Q(Q=self.Q_actions, beta=self.beta, epsilon=self.epsilon)
        self.action = np.random.choice(range(self.n_actions), p=self.p_actions)
        self.forget_Qs()  # Why is this happening here and not in learn?
        return self.action

    def learn(self, stimulus, action, reward):

        # Calculate RPEs
        self.RPEs_high = reward - self.Q_high
        self.RPEs_low = reward - self.Q_low[:, action]

        # Update Q values based on RPEs
        self.Q_low[:, action] += self.alpha * self.p_TS * self.RPEs_low
        self.Q_high += self.alpha_high * self.p_TS * self.RPEs_high

        # Calculate trial log likelihood and add to sum
        self.LL += np.log(self.p_actions[action])

    def get_p_from_Q(self, Q, beta, epsilon):

        # Softmax
        p_actions = np.empty(len(Q))
        for i in range(len(Q)):
            denominator = 1. + sum([np.exp(beta * (Q[j] - Q[i])) for j in range(len(Q)) if j != i])
            p_actions[i] = 1. / denominator

        # Add epsilon noise
        p_actions = epsilon / len(self.p_actions) + (1 - epsilon) * p_actions
        assert np.round(sum(p_actions), 3) == 1
        return p_actions

    def forget_Qs(self):
        self.Q_low -= self.forget * (self.Q_low - self.initial_Q)
        self.Q_high -= self.forget_high * (self.Q_high - self.initial_Q)
