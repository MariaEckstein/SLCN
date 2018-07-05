import numpy as np

import theano.tensor as T


class Shared(object):

    @staticmethod
    def get_paths():

        return {'human data': 'C:/Users/maria/MEGAsync/SLCN/PShumanData/',
                'fitting results': 'C:/Users/maria/MEGAsync/SLCN/PShumanData/fitting/',
                'simulations': 'C:/Users/maria/MEGAsync/SLCN/PSsimulations/',
                'old simulations': 'C:/Users/maria/MEGAsync/SLCN/PSGenRecCluster/fit_par/',
                'PS task info': 'C:/Users/maria/MEGAsync/SLCN/ProbabilisticSwitching/Prerandomized sequences/'}

    def p_from_Q(self, Q_left, Q_right, beta, epsilon):
        return 1 / (1 + np.exp(-beta * (Q_right - Q_left)))  # translate Q-values into probabilities using softmax

    def add_epsilon_noise(self, p_right, epsilon):
        return epsilon * 0.5 + (1 - epsilon) * p_right  # add epsilon noise

    def update_Q(self, reward, choice, Q_left, Q_right, alpha, calpha):

        # Counter-factual learning: Weigh RPE with alpha for chosen action, and with calpha for unchosen action
        RPE = reward - choice * Q_right - (1 - choice) * Q_left  # RPE = reward - Q[chosen]
        weight_left = (1 - choice) * alpha - choice * calpha  # choice==0: weight=alpha; choice==1; weight=-calpha
        weight_right = choice * alpha - (1 - choice) * calpha   # sim.
        return Q_left + weight_left * RPE, Q_right + weight_right * RPE

    def get_p_subsequent_trial(self, p_right, p_switch):

        # Take into account that a switch might occur
        return (1 - p_switch) * p_right + p_switch * (1 - p_right)

    def get_likelihoods(self, rewards, choices, p_reward, p_noisy_task):

        # Likelihood of outcome (reward 0 or 1) if choice was correct:
        #           |   reward==1   |   reward==0
        #           |---------------|-----------------
        # choice==1 | p_reward      | 1 - p_reward
        # choice==0 | p_noisy_task  | 1 - p_noisy_task

        lik_cor_rew1 = choices * p_reward + (1 - choices) * p_noisy_task
        lik_cor_rew0 = choices * (1 - p_reward) + (1 - choices) * (1 - p_noisy_task)
        lik_cor = rewards * lik_cor_rew1 + (1 - rewards) * lik_cor_rew0

        # Likelihood of outcome (reward 0 or 1) if choice was incorrect:
        #           |   reward==1   |   reward==0
        #           |---------------|-----------------
        # choice==1 | p_noisy_task  | 1 - p_noisy_task
        # choice==0 | p_reward      | 1 - p_reward

        lik_inc_rew1 = choices * p_noisy_task + (1 - choices) * p_reward
        lik_inc_rew0 = choices * (1 - p_noisy_task) + (1 - choices) * (1 - p_reward)
        lik_inc = rewards * lik_inc_rew1 + (1 - rewards) * lik_inc_rew0

        return lik_cor, lik_inc

    def post_from_lik(self, lik_cor, lik_inc, p_right):

        # Posterior probability that right action is correct, based on received feedback (likelihood)
        return lik_cor * p_right / (lik_cor * p_right + lik_inc * (1 - p_right))

    # def post_from_lik(self, lik_cor, lik_incor, choice, p_right):
    #
    #     # Prior: Get probability for the action that was taken in this trial instead of right action
    #     p_taken = p_right
    #     left_choice_idx = T.eq(choice, T.zeros_like(choice))
    #     p_taken = T.set_subtensor(p_taken[left_choice_idx], 1 - p_taken[left_choice_idx])
    #
    #     # Posterior: Calculate probability that the taken action was correct, based on received feedback (likelihood)
    #     post_taken = lik_cor * p_taken / (lik_cor * p_taken + lik_incor * (1 - p_taken))
    #
    #     # Get probabilities for the right action instead of trial's choice
    #     post_right = post_taken
    #     post_right = T.set_subtensor(post_right[left_choice_idx], 1 - post_right[left_choice_idx])
    #
    #     return post_right
    #
    # def post_from_lik_np(self, lik_cor, lik_incor, choice, p_right):
    #
    #     # Prior: Get probability for the action that was taken in this trial instead of right action
    #     p_taken = p_right
    #     left_choice_idx = np.equal(choice, np.zeros_like(choice))
    #     p_taken[left_choice_idx] = 1 - p_taken[left_choice_idx]
    #
    #     # Posterior: Calculate probability that the taken action was correct, based on received feedback (likelihood)
    #     post_taken = lik_cor * p_taken / (lik_cor * p_taken + lik_incor * (1 - p_taken))
    #
    #     # Get probabilities for the right action instead of trial's choice
    #     post_right = post_taken
    #     post_right[left_choice_idx] = 1 - post_right[left_choice_idx]
    #
    #     return post_right
    #
    # def get_likelihoods(self, rewards, p_reward, p_noisy_task):
    #     lik_cor = rewards * p_reward + (1 - rewards) * (1 - p_reward)
    #     lik_incor = rewards * p_noisy_task + (1 - rewards) * (1 - p_noisy_task)
    #     return lik_cor, lik_incor
