import numpy as np


class Shared(object):

    @staticmethod
    def get_paths():

        return {'human data': 'C:/Users/maria/MEGAsync/SLCN/PShumanData/',
                'fitting results': 'C:/Users/maria/MEGAsync/SLCN/PShumanData/fitting/',
                'simulations': 'C:/Users/maria/MEGAsync/SLCN/PSsimulations/',
                'old simulations': 'C:/Users/maria/MEGAsync/SLCN/PSGenRecCluster/fit_par/',
                'PS task info': 'C:/Users/maria/MEGAsync/SLCN/ProbabilisticSwitching/Prerandomized sequences/'}

    def p_from_Q(self, Q_left, Q_right, beta):
        return 1 / (1 + np.exp(-beta * (Q_right - Q_left)))  # translate Q-values into probabilities using softmax

    # def add_epsilon_noise(self, p_right, epsilon):
    # DON'T DO! IT NEEDS TO HAPPEN WITHIN THE SAME TRIAL, OTHERWISE IT'S WRONG!!
    #     return epsilon * 0.5 + (1 - epsilon) * p_right  # add epsilon noise

    def update_Q(self, reward, choice, Q_left, Q_right, alpha, calpha):

        # Counter-factual learning: Weigh RPE with alpha for chosen action, and with calpha for unchosen action
        RPE = reward - choice * Q_right - (1 - choice) * Q_left  # RPE = reward - Q[chosen]
        weight_left = (1 - choice) * alpha - choice * calpha  # choice==0: weight=alpha; choice==1; weight=-calpha
        weight_right = choice * alpha - (1 - choice) * calpha   # sim.
        return Q_left + weight_left * RPE, Q_right + weight_right * RPE

    # def get_p_subsequent_trial(self, p_right, p_switch):
    #
    #     # Take into account that a switch might occur
    #     return (1 - p_switch) * p_right + p_switch * (1 - p_right)

    def get_likelihoods(self, rewards, choices, p_reward, p_noisy_task):

        # p(r=r|choice=correct): Likelihood of outcome (reward 0 or 1) if choice was correct:
        #           |   reward==1   |   reward==0
        #           |---------------|-----------------
        # choice==1 | p_reward      | 1 - p_reward
        # choice==0 | p_noisy_task  | 1 - p_noisy_task

        lik_cor_rew1 = choices * p_reward + (1 - choices) * p_noisy_task
        lik_cor_rew0 = choices * (1 - p_reward) + (1 - choices) * (1 - p_noisy_task)
        lik_cor = rewards * lik_cor_rew1 + (1 - rewards) * lik_cor_rew0

        # p(r=r|choice=incorrect): Likelihood of outcome (reward 0 or 1) if choice was incorrect:
        #           |   reward==1   |   reward==0
        #           |---------------|-----------------
        # choice==1 | p_noisy_task  | 1 - p_noisy_task
        # choice==0 | p_reward      | 1 - p_reward

        lik_inc_rew1 = choices * p_noisy_task + (1 - choices) * p_reward
        lik_inc_rew0 = choices * (1 - p_noisy_task) + (1 - choices) * (1 - p_reward)
        lik_inc = rewards * lik_inc_rew1 + (1 - rewards) * lik_inc_rew0

        return lik_cor, lik_inc

    def post_from_lik(self, lik_cor, lik_inc, p_right, p_switch, epsilon):

        # Posterior probability that right action is correct, based on likelihood (i.e., received feedback)
        p_right = lik_cor * p_right / (lik_cor * p_right + lik_inc * (1 - p_right))

        # Take into account that a switch might occur
        p_right = (1 - p_switch) * p_right + p_switch * (1 - p_right)

        # Add epsilon noise
        return epsilon * 0.5 + (1 - epsilon) * p_right
