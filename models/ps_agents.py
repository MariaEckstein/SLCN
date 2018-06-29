import numpy as np


class RLAgent(object):

    def __init__(self, agent_stuff, all_pars, task_stuff=np.nan):

        # Get info about task
        self.n_actions, self.n_TS = task_stuff['n_actions'], agent_stuff['n_TS']

        # Get RL parameters
        self.learning_style = agent_stuff['learning_style']
        [self.alpha, self.alpha_high,
         self.beta, self.beta_high,
         self.epsilon, self.forget] = all_pars
        if self.alpha_high == 99:
            self.alpha_high = self.alpha
        if self.beta_high == 99:
            self.beta_high = self.beta
        self.forget_high = self.forget
        self.adjust_parameters(agent_stuff)

        # Set up value tables for TS values (Q_high) and action values (Q_low) for each TS
        self.initial_Q = 1. / self.n_actions
        self.Q_high = np.array([1., 0.])
        self.Q_low = self.initial_Q * np.ones([self.n_TS, self.n_actions])

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

        # Get betas on the right scale
        self.beta *= agent_stuff['beta_scaler']
        self.beta_high *= agent_stuff['beta_high_scaler']

        # Get the right high-level alpha and beta
        if self.alpha_high == 99:
            self.alpha_high = self.alpha
        if self.beta_high == 99:
            self.beta_high = self.beta

    def select_action(self):

        # Calculate probabilities for each TS from Q_high, and select action according to associated Q_low
        self.p_TS = self.get_p_from_Q(Q=self.Q_high, beta=self.beta_high, epsilon=0)
        self.Q_actions = np.dot(self.p_TS, self.Q_low)  # Weighted average
        self.p_actions = self.get_p_from_Q(Q=self.Q_actions, beta=self.beta, epsilon=self.epsilon)
        self.action = np.random.choice(range(self.n_actions), p=self.p_actions)
        self.forget_Qs()  # Why is this happening here and not in learn?
        return self.action

    def learn(self, action, reward):

        # Calculate trial log likelihood and add to sum
        self.LL += np.log(self.p_actions[action])

        # Calculate RPEs
        self.RPEs_high = reward - self.Q_high
        self.RPEs_low = reward - self.Q_low[:, action]

        # Update Q values based on RPEs
        update_high = self.alpha_high * self.p_TS * self.RPEs_high
        update_low = self.alpha * self.p_TS * self.RPEs_low
        self.Q_high += update_high
        self.Q_low[:, action] += update_low
        if 'counter' in self.learning_style:
            self.Q_low[:, 1-action] -= update_low

    def get_p_from_Q(self, Q, beta, epsilon):

        # Softmax
        p_actions = np.empty(len(Q))
        for i in range(len(Q)):
            denominator = 1. + sum([np.exp(beta * (Q[j] - Q[i])) for j in range(len(Q)) if j != i])
            p_actions[i] = 1. / denominator

        # Add epsilon noise and get probabilities into [0, 1] (minimizer will sometimes go outside)
        p_actions = epsilon / len(p_actions) + (1 - epsilon) * p_actions
        p_actions[np.argwhere(p_actions < 0)] = 0
        p_actions[np.argwhere(p_actions > 1)] = 1
        assert np.round(sum(p_actions), 3) == 1, 'Error in get_p_from_Q: probabilities must sum to 1.'
        assert np.all(p_actions >= 0), 'Error in get_p_from_Q: probabilities must be >= 0.'
        assert np.all(p_actions <= 1), 'Error in get_p_from_Q: probabilities must be <= 1.'
        return p_actions

    def forget_Qs(self):
        self.Q_low -= self.forget * (self.Q_low - self.initial_Q)
        self.Q_high -= self.forget_high * (self.Q_high - self.initial_Q)


class BayesAgent(object):
    def __init__(self, agent_stuff, all_pars, task_stuff):

        # Big general stuff
        self.learning_style = agent_stuff['learning_style']
        self.n_actions = task_stuff['n_actions']

        # Parameters and prior knowledge
        [_, _, _, _, self.epsilon, _] = all_pars
        self.p_switch = task_stuff['p_reward'] / np.mean(task_stuff['av_run_length'])  # true average switch probability
        self.p_reward = task_stuff['p_reward']  # true reward probability

        # "Memory"
        self.p_actions = np.ones(self.n_actions) / self.n_actions  # initialize prior uniformly over all actions
        self.LL = 0

        # Additional stuff for simulate_interactive
        self.Q_high = np.nan
        self.Q_low = np.full([2, 2], np.nan)
        self.p_TS = np.nan
        self.Q_actions = np.nan
        self.RPEs_high = np.nan
        self.RPEs_low = np.nan

    def select_action(self):
        return np.random.choice(range(self.n_actions), p=self.p_actions)

    def learn(self, action, reward):

        # Calculate trial log likelihood and add to sum
        self.LL += np.log(self.p_actions[action])

        # Get likelihood [P(r|non_chosen_box==magic), P(r|chosen_box=magic)]
        if reward:
            lik_boxes = np.zeros(self.n_actions) + 0.001  # P(r==1|non_chosen_box==magical) = 0
            lik_boxes[action] = self.p_reward  # P(r==1|chosen_box==magical) = p_reward
        else:
            lik_boxes = np.ones(self.n_actions) - 0.001  # P(r==0|non_chosen_box==magical) = 1
            lik_boxes[action] = 1 - self.p_reward  # P(r==0|chosen_box==magical) = 1 - p_reward

        # Get posterior [P(chosen_box==magical|r==r), P(non_chosen_box==magical|r==r)], with
        # P(box==magical|r==r) = P(r==r|box==magical) * P(box==magical) / P(r==r)
        posterior = lik_boxes * self.p_actions / np.sum(lik_boxes * self.p_actions)  # normalize such that sum == 1

        # Get probabilities for the next trial
        p_actions = (1 - self.p_switch) * posterior + self.p_switch * (1 - posterior)

        # Add epsilon noise
        p_actions = self.epsilon / len(p_actions) + (1 - self.epsilon) * p_actions

        # Make sure probabilities are in the right ranges
        p_actions[np.argwhere(p_actions < 0)] = 0
        p_actions[np.argwhere(p_actions > 1)] = 1
        assert np.round(sum(p_actions), 3) == 1, 'Error in get_p_from_Q: probabilities must sum to 1.'
        assert np.all(p_actions >= 0), 'Error in get_p_from_Q: probabilities must be >= 0.'
        assert np.all(p_actions <= 1), 'Error in get_p_from_Q: probabilities must be <= 1.'

        self.p_actions = p_actions

        # For interactive game
        self.Q_actions = lik_boxes
