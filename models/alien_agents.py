import numpy as np
import math


class Agent(object):

    def __init__(self, agent_stuff, all_pars, task_stuff=np.nan):

        # Get info about task
        self.n_actions, self.n_contexts, self.n_aliens, self.n_TS = task_stuff['n_actions'], task_stuff['n_contexts'], task_stuff['n_aliens'], agent_stuff['n_TS']

        # Get RL parameters
        [self.alpha, self.alpha_high,
         self.beta, self.beta_high,
         self.epsilon, self.forget, self.TS_bias] = all_pars
        self.forget_high = self.forget
        self.adjust_parameters(agent_stuff)

        # Set up value tables for context-TS associations (Q_high) and stimulus-action pairs (Q_low) for each TS
        self.Q_high = np.zeros([self.n_contexts+2,
                                self.n_TS+2])
        self.Q_low = np.zeros([self.n_contexts+2,
                               self.n_aliens,
                               self.n_actions])
        self.initial_Q = 5. / 3.

        # Initialize RL features and log likelihood (LL)
        self.p_TS = np.ones(self.n_TS) / self.n_TS  # P(TS|context)
        self.p_actions = np.ones(self.n_actions) / self.n_actions  # P(action|TS)
        self.Q_actions = np.nan
        self.RPEs_low = np.nan
        self.RPEs_high = np.nan
        self.LL = 0

        # Initialize agent's "memory"
        self.task_phase = np.nan
        self.action = []
        self.context_ext = 99
        self.context_int = np.nan
        self.seen_seasons = []

        # Stuff for competition phase
        self.p_stimuli = np.nan
        self.Q_stimuli = np.nan

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

        # Detect context switch, save current context
        context_switch = self.handle_context(stimulus)
        if context_switch:
            self.create_new_TS(self.context_int)

        # Calculate probabilities for each TS from Q_high, and select action according to associated Q_low
        self.p_TS = self.get_p_from_Q(Q=self.Q_high[self.context_int, :], beta=self.beta_high, epsilon=0)
        self.Q_actions = np.dot(self.p_TS, self.Q_low[0:len(self.p_TS), stimulus[1], :])  # Weighted average
        self.p_actions = self.get_p_from_Q(Q=self.Q_actions, beta=self.beta, epsilon=self.epsilon)
        self.action = np.random.choice(range(self.n_actions), p=self.p_actions)
        self.forget_Qs()  # Why is this happening here and not in learn?
        return self.action

    def handle_context(self, stimulus):

        # Detect context switch
        context_switch = self.context_ext != stimulus[0]

        # Set context_int (row that will be used and updated in Q_high) and context_ext (number the task sends)
        if self.task_phase == '2CloudySeason':
            self.context_int = self.n_contexts  # cloudy season -> new hidden context
        else:
            self.context_int = stimulus[0]  # rainbow season -> new rainbow context (see alien_task)
        self.context_ext = stimulus[0]
        return context_switch

    def create_new_TS(self, context):

        # Create new TS if context has never been seen before (InitialLearn, Refreshers, Cloudy(!), Rainbow)
        if context not in self.seen_seasons:
            self.init_TS()
            self.seen_seasons.append(context)

        # Reinitialize all TS in cloudy season
        elif self.task_phase in ['2CloudySeason']:
            self.Q_high[context, :] = self.initial_Q

    def init_TS(self):

        # Initialize a new column for the new TS in Q_high, and bias it for the new context
        self.Q_high[:, len(self.seen_seasons)] = self.initial_Q
        self.Q_high[len(self.seen_seasons), len(self.seen_seasons)] *= self.TS_bias

        # Initialize Q_low (no bias)
        self.Q_low[len(self.seen_seasons), :, :] = self.initial_Q

    def learn(self, stimulus, action, reward):

        # Update Q values
        if self.task_phase != '5RainbowSeason':  # no value updating in rainbow season!

            # Calculate RPEs
            self.RPEs_high = reward - self.Q_high[self.context_int, :]  # Q_high for all TSs, given context
            self.RPEs_low = reward - self.Q_low[:, stimulus[1], action]  # Q_low for all TSs, given alien & action

            # Update Q values based on RPEs
            self.Q_low[:, stimulus[1], action] += self.alpha * self.p_TS * self.RPEs_low
            self.Q_high[self.context_int, :] += self.alpha_high * self.p_TS * self.RPEs_high

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
        self.Q_low -= self.forget * (self.Q_low - 1)  # decays toward 1 (average of incorrect responses)
        self.Q_high -= self.forget_high * (self.Q_high - 1)  # decays toward 1 (average of incorrect responses)

    def competition_selection(self, stimuli, phase):
        self.Q_stimuli = [self.get_Q_for_stimulus(stimulus, phase) for stimulus in stimuli]
        self.p_stimuli = self.get_p_from_Q(self.Q_stimuli, self.beta, self.epsilon)
        selected_index = np.random.choice(range(len(stimuli)), p=self.p_stimuli)
        return stimuli[selected_index]

    def marginalize(self, Q, beta):
        # Calculates the weighted average of the entries in Q: \sum_{a_i} p(a_i) * Q(a_i)
        p = self.get_p_from_Q(Q, beta, self.epsilon)
        return np.dot(p, Q)

    def get_p_TSi(self):
        # \pi(TS_i) = \sum_{c_j} \pi(TS_i|c_j) p(c_j)
        p_TSi_given_cj = [self.get_p_from_Q(self.Q_high[c, :self.n_TS], self.beta_high, self.epsilon) for c in range(self.n_contexts)]
        return np.mean(p_TSi_given_cj, axis=0)

    def get_Q_for_stimulus(self, stimulus, phase):
        if phase == 'season':
            context = stimulus

            # Calculate "stimulus values": Q(c) = \sum_{TS_i} \pi(TS_i|c) Q(TS_i|c)
            Q_TSi_given_c = self.Q_high[context, :self.n_TS]
            return self.marginalize(Q_TSi_given_c, self.beta_high)

        elif phase == 'alien-same-season':
            context = stimulus[0]
            alien = stimulus[1]

            # Q(s|TS) = \sum_{a_i} \pi(a_i|s,TS) Q(a_i|s,TS)
            Q_ai_given_s_TSi = self.Q_low[:self.n_TS, alien, :]
            Q_s_given_TSi = [self.marginalize(Q_ai_given_s_TSi[TSi, :], self.beta) for TSi in range(self.n_TS)]

            # \pi(TS_i|c) = softmax(Q(TS_i|c))
            Q_TSi_given_c = self.Q_high[context, :self.n_TS]
            p_TSi_given_c = self.get_p_from_Q(Q_TSi_given_c, self.beta_high, self.epsilon)

            # Q(s,c) = \sum_{TS_i} \pi(TS_i|c) Q(s|TS_i)
            return np.dot(Q_s_given_TSi, p_TSi_given_c)

        elif phase == 'item':
            item = stimulus

            # Q(a|TS) = \sum_{s_i} \pi(a|s_i,TS) Q(a|s_i,TS)
            Q_a_given_s_TS = self.Q_low[:self.n_TS, :, item]
            Q_a_given_TS = [self.marginalize(Q_a_given_s_TS[TSi, :], self.beta) for TSi in range(self.n_TS)]

            # Q(a) = \sum_{TS_i} Q(a|TS_i) \pi(TS_i)
            p_TSi = self.get_p_TSi()
            return np.dot(Q_a_given_TS, p_TSi)

        elif phase == 'alien':
            alien = stimulus

            # Q(s|TS) = \sum_{a_i} \pi(a_i|s,TS) Q(a_i|s,TS)
            Q_a_given_s_TS = self.Q_low[:self.n_TS, alien, :]
            Q_s_given_TS = [self.marginalize(Q_a_given_s_TS[TSi, :], beta=self.beta) for TSi in range(self.n_TS)]

            # Q(s) = \sum_{TS_i} Q(s|TS_i) \pi(TS_i)
            p_TSi = self.get_p_TSi()
            return np.dot(Q_s_given_TS, p_TSi)
