import numpy as np
import theano
import theano.tensor as T


alien_initial_Q = 5 / 3


def get_paths(run_on_cluster):

    if run_on_cluster:
        # base_path = '/home/bunge/maria/Desktop/'
        # return {'human data': base_path + '/PShumanData/',
        #         'fitting results': base_path + '/ProbSwitch/fitting/',
        #         'SLCN info': base_path + '/ProbSwitch/SLCNinfo2.csv',
        #         'simulations': base_path + 'ProbSwitch/PSsimulations/',
        #         'old simulations': base_path + '/PShumanData/fit_par/',
        #         'PS task info': base_path + '/ProbabilisticSwitching/Prerandomized sequences/'}
        base_path = '/global/home/users/mariaeckstein/'
        return {'human data': base_path + 'PShumanData/',
                'fitting results': base_path + '/ProbSwitch/fitting/',
                'SLCN info': base_path + '/ProbSwitch/SLCNinfo2.csv',
                'simulations': base_path + 'ProbSwitch/PSsimulations/',
                'old simulations': base_path + '/PShumanData/fit_par/',
                'PS task info': base_path + '/ProbabilisticSwitching/Prerandomized sequences/'}

    else:
        base_path = 'C:/Users/maria/MEGAsync/SLCN'
        return {'human data': base_path + 'data/ProbSwitch/',
                'fitting results': base_path + '/PShumanData/fitting/map_indiv/',
                'SLCN info': base_path + 'data/SLCNinfo2.csv',
                'PS reward versions': base_path + 'data/ProbSwitch_rewardversions.csv',
                'ages': base_path + 'data/ages.csv',
                'simulations': base_path + '/PSsimulations/',
                'old simulations': base_path + '/PSGenRecCluster/fit_par/',
                'PS task info': base_path + '/ProbabilisticSwitching/Prerandomized sequences/'}


def get_strat_Qs(n_trials, n_subj):

    """code strategy 'stay unless you failed to receive reward twice in a row for the same action.'"""

    Qs = np.zeros((n_trials, n_subj, 2, 2, 2, 2, 2))  # (..., prev_prev_choice, prev_prev_reward, prev_choice, prev_reward, choice)
    Qs[:, :, :, :, 1, 1, 1] = 1  # ...     R & 1 -> R
    Qs[:, :, :, :, 0, 1, 0] = 1  # ...     L & 1 -> L
    Qs[:, :, 1, 1, 1, 0, 1] = 1  # R & 1 & R & 0 -> R
    Qs[:, :, 1, 0, 1, 0, 0] = 1  # R & 0 & R & 0 -> L
    Qs[:, :, 0, 1, 1, 0, 0] = 1  # L & 1 & R & 0 -> L
    Qs[:, :, 0, 0, 1, 0, 0] = 1  # L & 0 & R & 0 -> L
    Qs[:, :, 1, 1, 0, 0, 1] = 1  # R & 1 & L & 0 -> R
    Qs[:, :, 1, 0, 0, 0, 1] = 1  # R & 0 & L & 0 -> R
    Qs[:, :, 0, 0, 0, 0, 1] = 1  # L & 0 & L & 0 -> R
    Qs[:, :, 0, 1, 0, 0, 0] = 1  # L & 1 & L & 0 -> L
    return Qs


def get_WSLS_Qs(n_trials, n_subj):

    """code strategy 'stay unless you failed to receive reward twice in a row for the same action.'"""

    Qs = np.zeros((n_trials, n_subj, 2, 2, 2))  # (..., prev_choice, prev_reward, choice)
    Qs[:, :, 1, 1, 1] = 1  # R & 1 -> R
    Qs[:, :, 0, 1, 0] = 1  # L & 1 -> L
    Qs[:, :, 1, 0, 0] = 1  # R & 0 -> L
    Qs[:, :, 0, 0, 1] = 1  # L & 0 -> R
    return Qs


def p_from_Q(
        Qs, persev_bonus,
        prev_prev_choice, prev_prev_reward,
        prev_choice, prev_reward,
        init_p, n_subj,
        beta):

    """This is the 'raw' function. Others are copied from this one."""

    # # Comment in for p_from_Q_0 (letter models)
    # index0 = T.arange(n_subj), 0
    # index1 = T.arange(n_subj), 1

    # # Comment in for p_from_Q_1 (WSLS, abS, etc.)
    # index0 = T.arange(n_subj, dtype='int32'), prev_choice, prev_reward, 0
    # index1 = T.arange(n_subj, dtype='int32'), prev_choice, prev_reward, 1

    # Comment in for p_from_Q_2 (strat, abSS, etc.)
    index0 = T.arange(n_subj, dtype='int32'), prev_prev_choice, prev_prev_reward, prev_choice, prev_reward, 0
    index1 = T.arange(n_subj, dtype='int32'), prev_prev_choice, prev_prev_reward, prev_choice, prev_reward, 1

    # Add perseverance bonus (not working)  # TD fix
    Qs_p = Qs  # + persev_bonus

    # softmax-transform Q-values into probabilities
    p_right = 1 / (1 + np.exp(beta * (Qs_p[index0] - Qs_p[index1])))  # 0 = left action; 1 = right action
    p_right = 0.0001 + 0.9998 * p_right  # make 0.0001 < p_right < 0.9999

    return p_right


def update_Q(
        prev_prev_choice, prev_prev_reward,
        prev_choice, prev_reward,
        choice, reward,
        Qs, _,
        alpha, nalpha, calpha, cnalpha, n_subj):

    """This is the 'raw' function. Others are copied from this one."""

    # # Comment in for update_Q_0 (letter models)
    # index = T.arange(n_subj), choice
    # cindex = T.arange(n_subj), 1 - choice

    # # Comment in for update_Q_1 (WSLS, abS, etc.)
    # index = T.arange(n_subj), prev_choice, prev_reward, choice  # action taken (e.g., left & reward -> left)
    # mindex = T.arange(n_subj), 1 - prev_choice, prev_reward, 1 - choice  # mirror action (e.g., right & reward -> right)
    # cindex = T.arange(n_subj), prev_choice, prev_reward, 1 - choice  # counterf. action (e.g., left & reward -> right)
    # cmindex = T.arange(n_subj), 1 - prev_choice, prev_reward, choice  # counterf. mir. ac. (e.g, right & reward -> left)

    # Comment in for update_Q_2 (strat, abSS, etc.)
    index = T.arange(n_subj), prev_prev_choice, prev_prev_reward, prev_choice, prev_reward, choice
    mindex = T.arange(n_subj), 1 - prev_prev_choice, prev_prev_reward, 1 - prev_choice, prev_reward, 1 - choice
    cindex = T.arange(n_subj), prev_prev_choice, prev_prev_reward, prev_choice, prev_reward, 1 - choice
    cmindex = T.arange(n_subj), 1 - prev_prev_choice, prev_prev_reward, 1 - prev_choice, prev_reward, choice

    # Get reward prediction errors (RPEs) for positive (reward == 1) and negative outcomes (reward == 0)
    RPE = (reward - Qs[index]) * reward
    nRPE = (reward - Qs[index]) * (1 - reward)

    # Update action taken
    Qs = T.set_subtensor(Qs[index],
                         Qs[index] + alpha * RPE + nalpha * nRPE)

    # Update counterfactual action
    Qs = T.set_subtensor(Qs[cindex],
                         Qs[cindex] - calpha * RPE - cnalpha * nRPE)

    # Update mirror action (comment out for letter models)
    Qs = T.set_subtensor(Qs[mindex],
                         Qs[mindex] + alpha * RPE + nalpha * nRPE)

    # Update counterfactual mirror action (comment out for letter models)
    Qs = T.set_subtensor(Qs[cmindex],
                         Qs[cmindex] - calpha * RPE - cnalpha * nRPE)

    return Qs, _


def p_from_Q_0(
        Qs, persev_bonus,
        prev_prev_choice, prev_prev_reward,
        prev_choice, prev_reward,
        init_p, n_subj,
        beta):

    # Comment in for p_from_Q_0 (letter models)
    index0 = T.arange(n_subj), 0
    index1 = T.arange(n_subj), 1

    # Add perseverance bonus (not working)  # TD fix
    Qs_p = Qs  # + persev_bonus

    # softmax-transform Q-values into probabilities
    p_right = 1 / (1 + np.exp(beta * (Qs_p[index0] - Qs_p[index1])))  # 0 = left action; 1 = right action
    p_right = 0.0001 + 0.9998 * p_right  # make 0.0001 < p_right < 0.9999

    return p_right


def p_from_Q_1(
        Qs, persev_bonus,
        prev_prev_choice, prev_prev_reward,
        prev_choice, prev_reward,
        init_p, n_subj,
        beta):

    # Comment in for p_from_Q_1 (WSLS, abS, etc.)
    index0 = T.arange(n_subj, dtype='int32'), prev_choice, prev_reward, 0
    index1 = T.arange(n_subj, dtype='int32'), prev_choice, prev_reward, 1

    # Add perseverance bonus (not working)  # TD fix
    Qs_p = Qs  # + persev_bonus

    # softmax-transform Q-values into probabilities
    p_right = 1 / (1 + np.exp(beta * (Qs_p[index0] - Qs_p[index1])))  # 0 = left action; 1 = right action
    p_right = 0.0001 + 0.9998 * p_right  # make 0.0001 < p_right < 0.9999

    return p_right


def p_from_Q_2(
        Qs, persev_bonus,
        prev_prev_choice, prev_prev_reward,
        prev_choice, prev_reward,
        init_p, n_subj,
        beta):

    # Comment in for p_from_Q_2 (strat, abSS, etc.)
    index0 = T.arange(n_subj, dtype='int32'), prev_prev_choice, prev_prev_reward, prev_choice, prev_reward, 0
    index1 = T.arange(n_subj, dtype='int32'), prev_prev_choice, prev_prev_reward, prev_choice, prev_reward, 1

    # Add perseverance bonus (not working)  # TD fix
    Qs_p = Qs  # + persev_bonus

    # softmax-transform Q-values into probabilities
    p_right = 1 / (1 + np.exp(beta * (Qs_p[index0] - Qs_p[index1])))  # 0 = left action; 1 = right action
    p_right = 0.0001 + 0.9998 * p_right  # make 0.0001 < p_right < 0.9999

    return p_right


def p_from_Q_3(
        Qs, persev_bonus,
        prev_prev_prev_choice, prev_prev_prev_reward,
        prev_prev_choice, prev_prev_reward,
        prev_choice, prev_reward,
        init_p, n_subj,
        beta):

    # Comment in for p_from_Q_2 (strat, abSS, etc.)
    index0 = T.arange(n_subj, dtype='int32'), prev_prev_prev_choice, prev_prev_prev_reward, prev_prev_choice, prev_prev_reward, prev_choice, prev_reward, 0
    index1 = T.arange(n_subj, dtype='int32'), prev_prev_prev_choice, prev_prev_prev_reward, prev_prev_choice, prev_prev_reward, prev_choice, prev_reward, 1

    # Add perseverance bonus (not working)  # TD fix
    Qs_p = Qs  # + persev_bonus

    # softmax-transform Q-values into probabilities
    p_right = 1 / (1 + np.exp(beta * (Qs_p[index0] - Qs_p[index1])))  # 0 = left action; 1 = right action
    p_right = 0.0001 + 0.9998 * p_right  # make 0.0001 < p_right < 0.9999

    return p_right


def update_Q_0(
        prev_prev_choice, prev_prev_reward,
        prev_choice, prev_reward,
        choice, reward,
        Qs, _,
        alpha, nalpha, calpha, cnalpha, n_subj):

    # Comment in for update_Q_0 (letter models)
    index = T.arange(n_subj), choice
    cindex = T.arange(n_subj), 1 - choice

    # Get reward prediction errors (RPEs) for positive (reward == 1) and negative outcomes (reward == 0)
    RPE = (reward - Qs[index]) * reward
    nRPE = (reward - Qs[index]) * (1 - reward)

    # Update action taken
    Qs = T.set_subtensor(Qs[index],
                         Qs[index] + alpha * RPE + nalpha * nRPE)

    # Update counterfactual action
    Qs = T.set_subtensor(Qs[cindex],
                         Qs[cindex] - calpha * RPE - cnalpha * nRPE)

    return Qs, _


def update_Q_1(
        prev_prev_choice, prev_prev_reward,
        prev_choice, prev_reward,
        choice, reward,
        Qs, _,
        alpha, nalpha, calpha, cnalpha, n_subj):

    # Comment in for update_Q_1 (WSLS, abS, etc.)
    index = T.arange(n_subj), prev_choice, prev_reward, choice  # action taken (e.g., left & reward -> left)
    mindex = T.arange(n_subj), 1 - prev_choice, prev_reward, 1 - choice  # mirror action (e.g., right & reward -> right)
    cindex = T.arange(n_subj), prev_choice, prev_reward, 1 - choice  # counterf. action (e.g., left & reward -> right)
    cmindex = T.arange(n_subj), 1 - prev_choice, prev_reward, choice  # counterf. mir. ac. (e.g, right & reward -> left)

    # Get reward prediction errors (RPEs) for positive (reward == 1) and negative outcomes (reward == 0)
    RPE = (reward - Qs[index]) * reward
    nRPE = (reward - Qs[index]) * (1 - reward)

    # Update action taken
    Qs = T.set_subtensor(Qs[index],
                         Qs[index] + alpha * RPE + nalpha * nRPE)

    # Update counterfactual action
    Qs = T.set_subtensor(Qs[cindex],
                         Qs[cindex] - calpha * RPE - cnalpha * nRPE)

    # Update mirror action (comment out for letter models)
    Qs = T.set_subtensor(Qs[mindex],
                         Qs[mindex] + alpha * RPE + nalpha * nRPE)

    # Update counterfactual mirror action (comment out for letter models)
    Qs = T.set_subtensor(Qs[cmindex],
                         Qs[cmindex] - calpha * RPE - cnalpha * nRPE)

    return Qs, _


def update_Q_2(
        prev_prev_choice, prev_prev_reward,
        prev_choice, prev_reward,
        choice, reward,
        Qs, _,
        alpha, nalpha, calpha, cnalpha, n_subj):

    # Comment in for update_Q_2 (strat, abSS, etc.)
    index = T.arange(n_subj), prev_prev_choice, prev_prev_reward, prev_choice, prev_reward, choice
    mindex = T.arange(n_subj), 1 - prev_prev_choice, prev_prev_reward, 1 - prev_choice, prev_reward, 1 - choice
    cindex = T.arange(n_subj), prev_prev_choice, prev_prev_reward, prev_choice, prev_reward, 1 - choice
    cmindex = T.arange(n_subj), 1 - prev_prev_choice, prev_prev_reward, 1 - prev_choice, prev_reward, choice

    # Get reward prediction errors (RPEs) for positive (reward == 1) and negative outcomes (reward == 0)
    RPE = (reward - Qs[index]) * reward
    nRPE = (reward - Qs[index]) * (1 - reward)

    # Update action taken
    Qs = T.set_subtensor(Qs[index],
                         Qs[index] + alpha * RPE + nalpha * nRPE)

    # Update counterfactual action
    Qs = T.set_subtensor(Qs[cindex],
                         Qs[cindex] - calpha * RPE - cnalpha * nRPE)

    # Update mirror action (comment out for letter models)
    Qs = T.set_subtensor(Qs[mindex],
                         Qs[mindex] + alpha * RPE + nalpha * nRPE)

    # Update counterfactual mirror action (comment out for letter models)
    Qs = T.set_subtensor(Qs[cmindex],
                         Qs[cmindex] - calpha * RPE - cnalpha * nRPE)

    return Qs, _


def update_Q_3(
        prev_prev_prev_choice, prev_prev_prev_reward,
        prev_prev_choice, prev_prev_reward,
        prev_choice, prev_reward,
        choice, reward,
        Qs, _,
        alpha, nalpha, calpha, cnalpha, n_subj):

    # Comment in for update_Q_2 (strat, abSS, etc.)
    index = T.arange(n_subj), prev_prev_prev_choice, prev_prev_prev_reward, prev_prev_choice, prev_prev_reward, prev_choice, prev_reward, choice
    mindex = T.arange(n_subj), 1 - prev_prev_prev_choice, prev_prev_prev_reward, 1 - prev_prev_choice, prev_prev_reward, 1 - prev_choice, prev_reward, 1 - choice
    cindex = T.arange(n_subj), prev_prev_prev_choice, prev_prev_prev_reward, prev_prev_choice, prev_prev_reward, prev_choice, prev_reward, 1 - choice
    cmindex = T.arange(n_subj), 1 - prev_prev_prev_choice, prev_prev_prev_reward, 1 - prev_prev_choice, prev_prev_reward, 1 - prev_choice, prev_reward, choice

    # Get reward prediction errors (RPEs) for positive (reward == 1) and negative outcomes (reward == 0)
    RPE = (reward - Qs[index]) * reward
    nRPE = (reward - Qs[index]) * (1 - reward)

    # Update action taken
    Qs = T.set_subtensor(Qs[index],
                         Qs[index] + alpha * RPE + nalpha * nRPE)

    # Update counterfactual action
    Qs = T.set_subtensor(Qs[cindex],
                         Qs[cindex] - calpha * RPE - cnalpha * nRPE)

    # Update mirror action (comment out for letter models)
    Qs = T.set_subtensor(Qs[mindex],
                         Qs[mindex] + alpha * RPE + nalpha * nRPE)

    # Update counterfactual mirror action (comment out for letter models)
    Qs = T.set_subtensor(Qs[cmindex],
                         Qs[cmindex] - calpha * RPE - cnalpha * nRPE)

    return Qs, _


def update_Q_sim(
        prev_choice, prev_reward,
        choice, reward,
        Qs, _,
        alpha, nalpha, calpha, cnalpha, n_subj):

    index = np.arange(n_subj), prev_choice, prev_reward, choice  # action taken (e.g., left & reward -> left)
    mindex = np.arange(n_subj), 1 - prev_choice, prev_reward, 1 - choice  # mirror action (e.g., right & reward -> right)
    cindex = np.arange(n_subj), prev_choice, prev_reward, 1 - choice  # counterf. action (e.g., left & reward -> right)
    cmindex = np.arange(n_subj), 1 - prev_choice, prev_reward, choice  # counterf. mir. ac. (e.g, right & reward -> left)

    # Get reward prediction errors (RPEs) for positive (reward == 1) and negative outcomes (reward == 0)
    RPE = (reward - Qs[index]) * reward
    nRPE = (reward - Qs[index]) * (1 - reward)

    # Update action taken
    Qs[index] += alpha * RPE + nalpha * nRPE

    # Update mirror action
    Qs[mindex] += alpha * RPE + nalpha * nRPE

    # Update counterfactual action
    Qs[cindex] += - calpha * RPE - cnalpha * nRPE

    # Update counterfactual mirror action
    Qs[cmindex] += - calpha * RPE - cnalpha * nRPE

    return Qs, _

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


def post_from_lik(lik_cor, lik_inc, scaled_persev_bonus,
                  p_right,
                  p_switch, eps, beta, verbose=False):

    # Apply Bayes rule: Posterior prob. that right action is correct, based on likelihood (i.e., received feedback)
    p_right = lik_cor * p_right / (lik_cor * p_right + lik_inc * (1 - p_right))
    if verbose:
        print('p_right: {0} (after integrating prior)'.format(p_right.round(3)))

    # Take into account that a switch might occur
    p_right = (1 - p_switch) * p_right + p_switch * (1 - p_right)
    if verbose:
        print('p_right: {0} (after taking switch into account)'.format(p_right.round(3)))

    # Add perseverance bonus
    p_choice = p_right + scaled_persev_bonus
    if verbose:
        print('p_choice: {0} (after adding perseveration bonus)'.format(p_choice.round(3)))

    # Log-transform probabilities
    p_choice = 1 / (1 + np.exp(-beta * (p_choice - (1 - p_choice))))
    if verbose:
        print('p_choice: {0} (after sigmoid transform)'.format(p_choice.round(3)))

    # Add epsilon noise
    p_choice = eps * 0.5 + (1 - eps) * p_choice
    if verbose:
        print('p_choice: {0} (after adding epsilon noise)'.format(p_choice.round(3)))

    return p_right, p_choice


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