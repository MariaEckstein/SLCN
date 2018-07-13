import numpy as np


def get_paths(run_on_cluster):

    if not run_on_cluster:
        base_path = 'C:/Users/maria/MEGAsync/SLCN'
        return {'human data': base_path + '/PShumanData/',
                'fitting results': base_path + '/PShumanData/fitting/',
                'ages file name': base_path + 'data/SLCNinfo.csv',
                'simulations': base_path + '/PSsimulations/',
                'old simulations': base_path + '/PSGenRecCluster/fit_par/',
                'PS task info': base_path + '/ProbabilisticSwitching/Prerandomized sequences/'}

    else:
        base_path = '/home/bunge/maria/Desktop/'
        return {'human data': base_path + '/PShumanData/',
                'fitting results': base_path + '/PSPyMC3/fitting/',
                'ages file name': base_path + '/PSPyMC3/SLCNinfo.csv',
                'simulations': base_path + 'PSPyMC3/PSsimulations/',
                'old simulations': base_path + '/PShumanData/fit_par/',
                'PS task info': base_path + '/ProbabilisticSwitching/Prerandomized sequences/'}


def p_from_Q(Q_left, Q_right, beta, epsilon):

    # translate Q-values into probabilities using softmax
    p_right = 1 / (1 + np.exp(-beta * (Q_right - Q_left)))

    # add epsilon noise
    return epsilon * 0.5 + (1 - epsilon) * p_right


def update_Q(reward, choice, Q_left, Q_right, alpha, calpha):

    # Counter-factual learning: Weigh RPE with alpha for chosen action, and with calpha for unchosen action
    RPE = reward - choice * Q_right - (1 - choice) * Q_left  # RPE = reward - Q[chosen]
    weight_right = choice * alpha - (1 - choice) * calpha   # choice==0: weight=alpha; choice==1; weight=-calpha
    weight_left = (1 - choice) * alpha - choice * calpha  # sim.

    # Update Q-values for left and right action
    return Q_left + weight_left * RPE, Q_right + weight_right * RPE


def get_likelihoods(rewards, choices, p_reward, p_noisy):

    # p(r=r|choice=correct): Likelihood of outcome (reward 0 or 1) if choice was correct:
    #           |   reward==1   |   reward==0
    #           |---------------|-----------------
    # choice==1 | p_reward      | 1 - p_reward
    # choice==0 | p_noisy       | 1 - p_noisy

    lik_cor_rew1 = choices * p_reward + (1 - choices) * p_noisy
    lik_cor_rew0 = choices * (1 - p_reward) + (1 - choices) * (1 - p_noisy)
    lik_cor = rewards * lik_cor_rew1 + (1 - rewards) * lik_cor_rew0

    # p(r=r|choice=incorrect): Likelihood of outcome (reward 0 or 1) if choice was incorrect:
    #           |   reward==1   |   reward==0
    #           |---------------|-----------------
    # choice==1 | p_noisy       | 1 - p_noisy
    # choice==0 | p_reward      | 1 - p_reward

    lik_inc_rew1 = choices * p_noisy + (1 - choices) * p_reward
    lik_inc_rew0 = choices * (1 - p_noisy) + (1 - choices) * (1 - p_reward)
    lik_inc = rewards * lik_inc_rew1 + (1 - rewards) * lik_inc_rew0

    return lik_cor, lik_inc


def post_from_lik(lik_cor, lik_inc, p_right, p_switch, epsilon):

    # Posterior probability that right action is correct, based on likelihood (i.e., received feedback)
    p_right = lik_cor * p_right / (lik_cor * p_right + lik_inc * (1 - p_right))

    # Take into account that a switch might occur
    p_right = (1 - p_switch) * p_right + p_switch * (1 - p_right)

    # Add epsilon noise
    return epsilon * 0.5 + (1 - epsilon) * p_right
