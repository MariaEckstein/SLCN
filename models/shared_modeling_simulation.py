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


def get_WSLSS_Qs(n_trials, n_subj):

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
        Qs,
        prev_prev_choice, prev_prev_reward,
        prev_choice, prev_reward,
        init_p,
        n_subj, beta, persev):

    """This is the 'raw' function. Others are copied from this one."""

    # Comment in for p_from_Q_0 (letter models)
    index0 = T.arange(n_subj, dtype='int32'), 0
    index1 = T.arange(n_subj, dtype='int32'), 1

    # Comment in for p_from_Q_1 (WSLS, abS, etc.)
    index0 = T.arange(n_subj, dtype='int32'), prev_choice, prev_reward, 0
    index1 = T.arange(n_subj, dtype='int32'), prev_choice, prev_reward, 1

    # Comment in for p_from_Q_2 (WSLSS, abSS, etc.)
    index0 = T.arange(n_subj, dtype='int32'), prev_prev_choice, prev_prev_reward, prev_choice, prev_reward, 0
    index1 = T.arange(n_subj, dtype='int32'), prev_prev_choice, prev_prev_reward, prev_choice, prev_reward, 1

    # Add perseverance bonus
    Qs0 = Qs[index0]
    Qs1 = Qs[index1]

    one = T.ones(1, dtype='int16')  # to avoid upcasting, which crashes theano.scan, e.g.:
    # upcast problem 1: (np.array(1, dtype='float32') + np.array(1, dtype='int32')).dtype >>> dtype('float64')
    # fix: (np.array(1, dtype='float32') + np.array(1, dtype='int16')).dtype >>> dtype('float32')
    # upcast problem 2: (1 - np.array(1, dtype='float32')).dtype >>> dtype('float64')
    # fix (np.array(1, dtype='int16') - np.array(1, dtype='float32')).dtype >>> dtype('float32')

    Qs1 = Qs1 + prev_choice * persev  # upcast problem 1
    Qs0 = Qs0 + (one - prev_choice) * persev  # upcast problem 2

    # softmax-transform Q-values into probabilities
    p_right = one / (one + np.exp(beta * (Qs0 - Qs1)))  # 0 = left action; 1 = right action
    p_right = 0.0001 + 0.9998 * p_right  # make 0.0001 < p_right < 0.9999

    return p_right


def update_Q(
        prev_prev_choice, prev_prev_reward,
        prev_choice, prev_reward,
        choice, reward,
        Qs, _,
        alpha, nalpha, calpha, cnalpha, n_subj):

    """This is the 'raw' function. Others are copied from this one."""

    # Comment in for update_Q_0 (letter models)
    index = T.arange(n_subj), choice
    cindex = T.arange(n_subj), 1 - choice

    # Comment in for update_Q_1 (WSLS, abS, etc.)
    index = T.arange(n_subj), prev_choice, prev_reward, choice  # action taken (e.g., left & reward -> left)
    mindex = T.arange(n_subj), 1 - prev_choice, prev_reward, 1 - choice  # mirror action (e.g., right & reward -> right)
    cindex = T.arange(n_subj), prev_choice, prev_reward, 1 - choice  # counterf. action (e.g., left & reward -> right)
    cmindex = T.arange(n_subj), 1 - prev_choice, prev_reward, choice  # counterf. mir. ac. (e.g, right & reward -> left)

    # Comment in for update_Q_2 (WSLSS, abSS, etc.)
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
        Qs,
        prev_prev_choice, prev_prev_reward,
        prev_choice, prev_reward,
        init_p, n_subj,
        beta, persev):

    # Comment in for p_from_Q_0 (letter models)
    index0 = T.arange(n_subj, dtype='int32'), 0
    index1 = T.arange(n_subj, dtype='int32'), 1

    # Add perseverance bonus
    Qs0 = Qs[index0]
    Qs1 = Qs[index1]

    one = T.ones(1, dtype='int16')  # to avoid upcasting, which crashes theano.scan, e.g.:

    Qs1 = Qs1 + prev_choice * persev
    Qs0 = Qs0 + (one - prev_choice) * persev

    # softmax-transform Q-values into probabilities
    p_right = one / (one + np.exp(beta * (Qs0 - Qs1)))  # 0 = left action; 1 = right action
    p_right = 0.0001 + 0.9998 * p_right  # make 0.0001 < p_right < 0.9999

    return p_right


def p_from_Q_1(
        Qs,
        prev_prev_choice, prev_prev_reward,
        prev_choice, prev_reward,
        init_p, n_subj,
        beta, persev):

    # Comment in for p_from_Q_1 (WSLS, abS, etc.)
    index0 = T.arange(n_subj, dtype='int32'), prev_choice, prev_reward, 0
    index1 = T.arange(n_subj, dtype='int32'), prev_choice, prev_reward, 1

    # Add perseverance bonus
    Qs0 = Qs[index0]
    Qs1 = Qs[index1]

    one = T.ones(1, dtype='int16')  # to avoid upcasting, which crashes theano.scan, e.g.:

    Qs1 = Qs1 + prev_choice * persev
    Qs0 = Qs0 + (one - prev_choice) * persev

    # softmax-transform Q-values into probabilities
    p_right = one / (one + np.exp(beta * (Qs0 - Qs1)))  # 0 = left action; 1 = right action
    p_right = 0.0001 + 0.9998 * p_right  # make 0.0001 < p_right < 0.9999

    return p_right


def p_from_Q_2(
        Qs,
        prev_prev_choice, prev_prev_reward,
        prev_choice, prev_reward,
        init_p, n_subj,
        beta, persev):

    # Comment in for p_from_Q_2 (WSLSS, abSS, etc.)
    index0 = T.arange(n_subj, dtype='int32'), prev_prev_choice, prev_prev_reward, prev_choice, prev_reward, 0
    index1 = T.arange(n_subj, dtype='int32'), prev_prev_choice, prev_prev_reward, prev_choice, prev_reward, 1

    # Add perseverance bonus
    Qs0 = Qs[index0]
    Qs1 = Qs[index1]

    one = T.ones(1, dtype='int16')  # to avoid upcasting, which crashes theano.scan, e.g.:

    Qs1 = Qs1 + prev_choice * persev
    Qs0 = Qs0 + (one - prev_choice) * persev

    # softmax-transform Q-values into probabilities
    p_right = one / (one + np.exp(beta * (Qs0 - Qs1)))  # 0 = left action; 1 = right action
    p_right = 0.0001 + 0.9998 * p_right  # make 0.0001 < p_right < 0.9999

    return p_right


def p_from_Q_3(
        Qs,
        prev_prev_prev_choice, prev_prev_prev_reward,
        prev_prev_choice, prev_prev_reward,
        prev_choice, prev_reward,
        init_p, n_subj,
        beta, persev):

    """Draft - not tested"""

    # Comment in for p_from_Q_2 (WSLSS, abSS, etc.)
    index0 = T.arange(n_subj, dtype='int32'), prev_prev_prev_choice, prev_prev_prev_reward, prev_prev_choice, prev_prev_reward, prev_choice, prev_reward, 0
    index1 = T.arange(n_subj, dtype='int32'), prev_prev_prev_choice, prev_prev_prev_reward, prev_prev_choice, prev_prev_reward, prev_choice, prev_reward, 1

    # Add perseverance bonus
    Qs0 = Qs[index0]
    Qs1 = Qs[index1]

    one = T.ones(1, dtype='int16')  # to avoid upcasting, which crashes theano.scan, e.g.:

    Qs1 = Qs1 + prev_choice * persev
    Qs0 = Qs0 + (one - prev_choice) * persev

    # softmax-transform Q-values into probabilities
    p_right = one / (one + np.exp(beta * (Qs0 - Qs1)))  # 0 = left action; 1 = right action
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

    # Comment in for update_Q_2 (WSLSS, abSS, etc.)
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

    # Comment in for update_Q_2 (WSLSS, abSS, etc.)
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
        prev_prev_choice, prev_prev_reward,
        prev_choice, prev_reward,
        choice, reward,
        Qs, _,
        alpha, nalpha, calpha, cnalpha,
        n_subj, n_trials_back, verbose):

    """Should be blueprint for all the copies for the theano function."""

    if n_trials_back == 0:
        index = np.arange(n_subj), choice
        cindex = np.arange(n_subj), 1 - choice

    elif n_trials_back == 1:
        index = np.arange(n_subj), prev_choice, prev_reward, choice  # action taken (e.g., left & reward -> left)
        mindex = np.arange(n_subj), 1 - prev_choice, prev_reward, 1 - choice  # mirror action (e.g., right & reward -> right)
        cindex = np.arange(n_subj), prev_choice, prev_reward, 1 - choice  # counterf. action (e.g., left & reward -> right)
        cmindex = np.arange(n_subj), 1 - prev_choice, prev_reward, choice  # counterf. mir. ac. (e.g, right & reward -> left)

    elif n_trials_back == 2:
        index = np.arange(n_subj), prev_prev_choice, prev_prev_reward, prev_choice, prev_reward, choice  # action taken (e.g., left & reward -> left)
        mindex = np.arange(n_subj), 1 - prev_prev_choice, prev_prev_reward, 1 - prev_choice, prev_reward, 1 - choice  # mirror action (e.g., right & reward -> right)
        cindex = np.arange(n_subj), prev_prev_choice, prev_prev_reward, prev_choice, prev_reward, 1 - choice  # counterf. action (e.g., left & reward -> right)
        cmindex = np.arange(n_subj), 1 - prev_prev_choice, prev_prev_reward, 1 - prev_choice, prev_reward, choice  # counterf. mir. ac. (e.g, right & reward -> left)

    # Get reward prediction errors (RPEs) for positive (reward == 1) and negative outcomes (reward == 0)
    RPE = (reward - Qs[index]) * reward
    nRPE = (reward - Qs[index]) * (1 - reward)

    # Update action taken
    Qs[index] += alpha * RPE + nalpha * nRPE

    # Update counterfactual action
    Qs[cindex] += - calpha * RPE - cnalpha * nRPE

    if n_trials_back > 0:
        # Update mirror action
        Qs[mindex] += alpha * RPE + nalpha * nRPE

        # Update counterfactual mirror action
        Qs[cmindex] += - calpha * RPE - cnalpha * nRPE

    if verbose:
        print('upd_Q - index: ', index)
        print('upd_Q - cindex: ', cindex)
        print('upd_Q - RPE: ', np.round(RPE, 2))
        print('upd_Q - nRPE: ', np.round(nRPE, 2))
        print('upd_Q - new Qs:\n', np.round(Qs, 2))

    return Qs, _


def p_from_Q_sim(
        Qs,
        prev_prev_choice, prev_prev_reward,
        prev_choice, prev_reward,
        init_p, n_subj,
        beta, persev,
        n_trials_back, verbose):

    """Should be the blueprint for the theano functions."""

    if n_trials_back == 0:
        index0 = np.arange(n_subj, dtype='int32'), 0
        index1 = np.arange(n_subj, dtype='int32'), 1

    elif n_trials_back == 1:
        index0 = np.arange(n_subj, dtype='int32'), prev_choice, prev_reward, 0
        index1 = np.arange(n_subj, dtype='int32'), prev_choice, prev_reward, 1

    elif n_trials_back == 2:
        index0 = np.arange(n_subj, dtype='int32'), prev_prev_choice, prev_prev_reward, prev_choice, prev_reward, 0
        index1 = np.arange(n_subj, dtype='int32'), prev_prev_choice, prev_prev_reward, prev_choice, prev_reward, 1

    # Add perseverance bonus
    Qs0 = Qs[index0]
    Qs1 = Qs[index1]

    Qs1 += prev_choice * persev
    Qs0 += (1 - prev_choice) * persev

    # softmax-transform Q-values into probabilities
    p_right = 1 / (1 + np.exp(beta * (Qs0 - Qs1)))  # 0 = left action; 1 = right action
    p_right = 0.0001 + 0.9998 * p_right  # make 0.0001 < p_right < 0.9999

    if verbose:
        print('p_Q - beta: ', np.round(beta, 2))
        print('p_Q - index0: ', index0)
        print('p_Q - index1: ', index1)
        print('p_Q - Q0:\n', np.round(Qs0, 2))
        print('p_Q - Q1:\n', np.round(Qs1, 2))
        print('p_Q - p_right: ', np.round(p_right, 2))

    return p_right


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
                  p_switch, beta, verbose=False):

    if verbose:
        print('old p_right:\n{0}'.format(p_right.round(3)))

    # Apply Bayes rule: Posterior prob. that right action is correct, based on likelihood (i.e., received feedback)
    p_right = lik_cor * p_right / (lik_cor * p_right + lik_inc * (1 - p_right))
    # theano.printing.Print('p_right after integr prior')(p_right)
    if verbose:
        print('p_right after likelihood:\n{0}'.format(p_right.round(3)))

    # Take into account that a switch might occur
    p_right = (1 - p_switch) * p_right + p_switch * (1 - p_right)
    # theano.printing.Print('p_right after taking switch')(p_right)
    if verbose:
        print('p_right after taking switch into account:\n{0}'.format(p_right.round(3)))

    # Add perseverance bonus
    p_choice = p_right + scaled_persev_bonus
    # theano.printing.Print('p_right after adding persevation bonus')(p_choice)
    if verbose:
        print('p_choice after adding perseveration bonus:\n{0}'.format(p_choice.round(3)))

    # Log-transform probabilities
    p_choice = 1 / (1 + np.exp(-beta * (p_choice - (1 - p_choice))))
    p_choice = 0.0001 + 0.9998 * p_choice  # make 0.0001 < p_choice < 0.9999
    # theano.printing.Print('p_right after softmax')(p_choice)
    if verbose:
        print('p_choice after sigmoid transform:\n{0}'.format(p_choice.round(3)))

    return p_right, p_choice


def get_n_trials_back(model_name):
    if 'SSS' in model_name:  # abSSS, etc.
        return 3
    elif 'SS' in model_name:  # WSLSS, abSS, etc.
        return 2
    elif 'S' in model_name:  # WSLS and abS, abcS, etc.
        return 1
    else:  # ab, abc, etc.; Bayes models
        return 0


def get_n_params(model_name):
    if all(i in model_name for i in 'abcnp') and 'nc' in model_name:  # abcnncp, abcnncpS, abcnncpSS, etc.
        return 6
    elif all(i in model_name for i in 'abcn') and 'cn' in model_name or all(i in model_name for i in 'abcnp'):  # abcnnc, abcnncS, abcnncSS, etc.
        return 5
    elif all(i in model_name for i in 'abcn') or all(i in model_name for i in 'abpn') or all(i in model_name for i in 'abcp') or all(i in model_name for i in 'bpsr'):
        return 4
    elif all(i in model_name for i in 'abc') or all(i in model_name for i in 'abn') or all(i in model_name for i in 'abp') or all(i in model_name for i in 'bsr') or all(i in model_name for i in 'bpr') or all(i in model_name for i in 'bsp'):
        return 3
    elif 'ab' in model_name or all(i in model_name for i in 'bs') or all(i in model_name for i in 'br') or all(i in model_name for i in 'bp'):
        return 2
    elif 'WSLS' in model_name or 'b' in model_name:  # WSLS, WSLSS, Bayesian b model
        return 1  # just beta
    else:
        raise ValueError("I don't have n_params stored for this model. Please add your model to `get_n_params`.")



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