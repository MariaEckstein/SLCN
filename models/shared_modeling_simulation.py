import numpy as np

import pymc3 as pm
import theano
import theano.tensor as T


alien_initial_Q = 5 / 3

def get_paths(run_on_cluster):

    if run_on_cluster:
        base_path = '/home/bunge/maria/Desktop/'
        return {'human data': base_path + '/PShumanData/',
                'fitting results': base_path + '/PSPyMC3/fitting/',
                'ages file name': base_path + '/PSPyMC3/SLCNinfo.csv',
                'simulations': base_path + 'PSPyMC3/PSsimulations/',
                'old simulations': base_path + '/PShumanData/fit_par/',
                'PS task info': base_path + '/ProbabilisticSwitching/Prerandomized sequences/'}

    else:
        base_path = 'C:/Users/maria/MEGAsync/SLCN'
        return {'human data': base_path + '/PShumanData/',
                'fitting results': base_path + '/PShumanData/fitting/',
                'ages file name': base_path + 'data/SLCNinfo.csv',
                'fitted ages file name': base_path + 'data/ages.csv',
                'simulations': base_path + '/PSsimulations/',
                'old simulations': base_path + '/PSGenRecCluster/fit_par/',
                'PS task info': base_path + '/ProbabilisticSwitching/Prerandomized sequences/'}


def get_alien_paths(run_on_cluster):

    if run_on_cluster:
        base_path = '/home/bunge/maria/Desktop/'
        return {'human data': base_path + '/AlienshumanData/',
                'fitting results': base_path + '/AliensPyMC3/fitting/',
                'simulations': base_path + 'AliensPyMC3/PSsimulations/'}

    else:
        base_path = 'C:/Users/maria/MEGAsync/Berkeley/TaskSets'
        return {'human data': base_path + '/Data/version3.1/',
                'fitting results': base_path + '/Data/version3.1/fitting/',
                'simulations': base_path + '/Data/simulations/'}


def p_from_Q(Q_left, Q_right, beta, eps):

    # translate Q-values into probabilities using softmax
    p_right = 1 / (1 + np.exp(-beta * (Q_right - Q_left)))

    # add eps noise
    return eps * 0.5 + (1 - eps) * p_right


# def p_from_Q_aliens(Q0, Q1, Q2, beta):

    # # translate Q-values into probabilities using softmax
    # Q0_exp = np.exp(beta * Q0)
    # Q1_exp = np.exp(beta * Q1)
    # Q2_exp = np.exp(beta * Q2)
    #
    # Qsum_exp = Q0_exp + Q1_exp + Q2_exp
    #
    # p0 = Q0_exp / Qsum_exp
    # p1 = Q1_exp / Qsum_exp
    # p2 = Q2_exp / Qsum_exp

def TS_from_Q_high(Q, beta):

    # Q_exp = np.exp(beta * Q[season, :])
    # T.printing.Print('Q_exp')(Q_exp)

    Q0_exp = np.exp(beta * Q[0])
    Q1_exp = np.exp(beta * Q[1])
    Q2_exp = np.exp(beta * Q[2])

    # Qsum_exp = T.sum(Q_exp, axis=1)

    Qsum_exp = Q0_exp + Q1_exp + Q2_exp

    p0 = Q0_exp / Qsum_exp
    p1 = Q1_exp / Qsum_exp
    p2 = Q2_exp / Qsum_exp

    p = T.as_tensor_variable([p0, p1, p2])

    return pm.Categorical('TS', p=p)


def action_from_Q_low(Q, beta, action):

    # Q_exp = np.exp(beta * Q[season, :])
    # T.printing.Print('Q_exp')(Q_exp)

    T.printing.Print('Q_low')(Q)
    T.printing.Print('beta')(beta)
    T.printing.Print('action')(action)

    Q0_exp = np.exp(beta * Q[0])
    Q1_exp = np.exp(beta * Q[1])
    Q2_exp = np.exp(beta * Q[2])

    # Qsum_exp = T.sum(Q_exp, axis=1)

    Qsum_exp = Q0_exp + Q1_exp + Q2_exp

    p0 = Q0_exp / Qsum_exp
    p1 = Q1_exp / Qsum_exp
    p2 = Q2_exp / Qsum_exp

    p = T.as_tensor_variable([p0, p1, p2])
    T.printing.Print('p_low')(p)

    return pm.Categorical('act', p=p, observed=action)


def update_Q_aliens(season, alien, action, reward,  # sequences
                    Q_low, Q_high,  # output-info
                    alpha_low, alpha_high, beta_low, beta_high, n_subj  # non-sequences
                    ):

    # Select TS
    TS, _ = theano.scan(fn=TS_from_Q_high,
                        sequences=[Q_high[T.arange(n_subj), season], beta_high])

    # Calculate RPEs at both levels
    RPE_high = reward - Q_high[T.arange(n_subj), season, TS]
    RPE_low = reward - Q_low[T.arange(n_subj), TS, alien, action]

    # Update Q-values based on RPEs
    T.set_subtensor(Q_high[T.arange(n_subj), season, TS],
                    Q_high[T.arange(n_subj), season, TS] + alpha_high * RPE_high)
    T.set_subtensor(Q_low[T.arange(n_subj), TS, alien, action],
                    Q_low[T.arange(n_subj), TS, alien, action] + alpha_low * RPE_low)

    T.printing.Print('RPE_high')(RPE_high)
    T.printing.Print('RPE_low')(RPE_low)

    # Select action based on Q_low
    action, _ = theano.scan(fn=action_from_Q_low,
                            sequences=[Q_low[T.arange(n_subj), TS, alien, :], beta_low, action])


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

    # Posterior probability that right action is correct, based on likelihood (i.e., received feedback)
    p_right = lik_cor * p_right / (lik_cor * p_right + lik_inc * (1 - p_right))
    if verbose:
        print('p_right: {0} (after applying Bayes rule)'.format(p_right.round(3)))

    # Take into account that a switch might occur
    p_right = (1 - p_switch) * p_right + p_switch * (1 - p_right)
    if verbose:
        print('p_right: {0} (after taking switch into account)'.format(p_right.round(3)))

    # Log-transform probabilities
    # p_right = 1 / (1 + np.exp(-beta * (p_right - (1 - p_right))))
    if verbose:
        print('p_right: {0} (after sigmoid transform)'.format(p_right.round(3)))

    # Add epsilon noise
    p_right = eps * 0.5 + (1 - eps) * p_right
    if verbose:
        print('p_right: {0} (after adding epsilon noise)'.format(p_right.round(3)))

    return p_right
