import numpy as np
import theano.tensor as T


alien_initial_Q = 5 / 3


def get_paths(run_on_cluster):

    if run_on_cluster:
        base_path = '/home/bunge/maria/Desktop/'
        return {'human data': base_path + '/PShumanData/',
                'fitting results': base_path + '/PSPyMC3/fitting/',
                'SLCN info': base_path + '/PSPyMC3/SLCNinfo2.csv',
                'simulations': base_path + 'PSPyMC3/PSsimulations/',
                'old simulations': base_path + '/PShumanData/fit_par/',
                'PS task info': base_path + '/ProbabilisticSwitching/Prerandomized sequences/'}

    else:
        base_path = 'C:/Users/maria/MEGAsync/SLCN'
        return {'human data': base_path + 'data/ProbSwitch/',
                'fitting results': base_path + '/PShumanData/fitting/',
                'SLCN info': base_path + 'data/SLCNinfo2.csv',
                'PS reward versions': base_path + 'data/ProbSwitch_rewardversions.csv',
                'ages': base_path + 'data/ages.csv',
                'simulations': base_path + '/PSsimulations/',
                'old simulations': base_path + '/PSGenRecCluster/fit_par/',
                'PS task info': base_path + '/ProbabilisticSwitching/Prerandomized sequences/'}


def get_alien_paths(run_on_cluster):

    if run_on_cluster:
        base_path = '/home/bunge/maria/Desktop/'
        return {'human data': base_path + '/AlienshumanData/',
                'fitting results': base_path + '/AliensPyMC3/fitting/',
                'simulations': base_path + 'AliensPyMC3/Aliensimulations/'}

    else:
        base_path = 'C:/Users/maria/MEGAsync/Berkeley/TaskSets'
        return {'human data': base_path + '/Data/version3.1/',
                'fitting results': base_path + '/Data/version3.1/fitting/',
                'simulations': 'C:/Users/maria/MEGAsync/SLCN/PSsimulations/'}


def p_from_Q(Q_left, Q_right, beta, eps):

    # translate Q-values into probabilities using softmax
    p_right = 1 / (1 + np.exp(-beta * (Q_right - Q_left)))

    # add eps noise
    return eps * 0.5 + (1 - eps) * p_right


def update_Q(reward, choice, Q_left, Q_right, alpha, nalpha, calpha, cnalpha):

    # Counter-factual learning: Weigh RPE with alpha for chosen action, and with calpha for unchosen action
    # Reward-sensitive learning: Different learning rates for positive (rew1) and negative (rew0) outcomes
    RPE = reward - choice * Q_right - (1 - choice) * Q_left  # RPE = reward - Q[chosen]
    alpha_right_rew1 = choice * alpha - (1 - choice) * calpha   # choice==0: weight=alpha; choice==1; weight=-calpha
    alpha_right_rew0 = choice * nalpha - (1 - choice) * cnalpha   # sim.

    alpha_left_rew1 = (1 - choice) * alpha - choice * calpha  # sim.
    alpha_left_rew0 = (1 - choice) * nalpha - choice * cnalpha  # sim.

    alpha_right = reward * alpha_right_rew1 + (1 - reward) * alpha_right_rew0
    alpha_left = reward * alpha_left_rew1 + (1 - reward) * alpha_left_rew0

    # Update Q-values for left and right action
    return Q_left + alpha_left * RPE, Q_right + alpha_right * RPE


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


def post_from_lik(lik_cor, lik_inc, p_right, p_switch, eps, beta, verbose=False):

    # Apply Bayes rule: Posterior prob. that right action is correct, based on likelihood (i.e., received feedback)
    p_right = lik_cor * p_right / (lik_cor * p_right + lik_inc * (1 - p_right))
    if verbose:
        print('p_right: {0} (after integrating prior)'.format(p_right.round(3)))

    # Take into account that a switch might occur
    p_right = (1 - p_switch) * p_right + p_switch * (1 - p_right)
    if verbose:
        print('p_right: {0} (after taking switch into account)'.format(p_right.round(3)))

    # Log-transform probabilities
    p_right = 1 / (1 + np.exp(-beta * (p_right - (1 - p_right))))
    if verbose:
        print('p_right: {0} (after sigmoid transform)'.format(p_right.round(3)))

    # Add epsilon noise
    p_right = eps * 0.5 + (1 - eps) * p_right
    if verbose:
        print('p_right: {0} (after adding epsilon noise)'.format(p_right.round(3)))

    return p_right


def softmax(X, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p