import numpy as np
import theano.tensor as T

from theano.tensor.shared_randomstreams import RandomStreams
rs = RandomStreams()


# Initial Q-value for actions and TS
alien_initial_Q = 1.2


def update_Qs_th_sim(season, alien,
              Q_low, Q_high,
              beta, beta_high, alpha, alpha_high, forget, forget_high, n_subj, n_TS, n_actions, def_TS):

    # Select TS
    # Q_high_sub = Q_high[T.arange(n_subj), season]  # Q_high_sub.shape -> [n_subj, n_TS]
    # p_high = T.nnet.softmax(beta_high * Q_high_sub)
    TS = season  # Flat
    # TS = T.argmax(Q_high_sub, axis=1)  # Hierarchical deterministic
    # rand = rs.uniform()
    # cumsum = T.extra_ops.cumsum(p_high, axis=1)
    # TS = n_TS - T.sum(rand < cumsum, axis=1)

    # Calculate action probabilities based on TS
    Q_low_sub = Q_low[T.arange(n_subj), TS, alien]  # Q_low_sub.shape -> [n_subj, n_actions]
    p_low = T.nnet.softmax(beta * Q_low_sub)
    rand = rs.uniform(size=(n_subj, 1))
    cumsum = T.extra_ops.cumsum(p_low, axis=1)
    action = n_actions - T.sum(rand < cumsum, axis=1)
    # action = [np.random.choice(range(n_actions), p=p_low_subj) for p_low_subj in p_low]
    reward = def_TS[season, alien, action]
    correct = reward > 1

    # # Add noise to reward
    # noised_reward = np.round(np.random.normal(reward, 0.05), 2)
    # noised_reward[noised_reward < 0] = 0

    # Forget Q-values a little bit
    Q_low = (1 - forget) * Q_low + forget * alien_initial_Q
    Q_high = (1 - forget_high) * Q_high + forget_high * alien_initial_Q

    # Calculate RPEs & update Q-values
    current_trial_high = T.arange(n_subj), season, TS
    RPE_high = reward - Q_high[current_trial_high]
    Q_high = T.set_subtensor(Q_high[current_trial_high],
                             Q_high[current_trial_high] + alpha_high * RPE_high)

    current_trial_low = T.arange(n_subj), TS, alien, action
    RPE_low = reward - Q_low[current_trial_low]
    Q_low = T.set_subtensor(Q_low[current_trial_low],
                            Q_low[current_trial_low] + alpha * RPE_low)

    return [Q_low, Q_high, TS, action, reward, correct, p_low]

# Same, but without theano, and selecting actions rather than reading them in from a file
def update_Qs_sim(season, alien,
                  Q_low, Q_high,
                  beta, beta_high, alpha, alpha_high, forget, forget_high,
                  n_subj, n_actions, n_TS, task, verbose=False):

    # Select TS
    Q_high_sub = Q_high[np.arange(n_subj), season]  # Q_high_sub.shape -> (n_subj, n_TS)
    p_high = softmax(beta_high * Q_high_sub, axis=1)   # p_high.shape -> (n_subj, n_TS)
    # TS = season  # Flat
    # TS = 0  # fs
    # TS = Q_high_sub.argmax(axis=1)  # Hierarchical deterministic
    TS = np.array([np.random.choice(a=n_TS, p=p_high_subj) for p_high_subj in p_high])  # Hierarchical softmax

    # Calculate action probabilities based on TS and select action
    Q_low_sub = Q_low[np.arange(n_subj), TS, alien]  # Q_low_sub.shape -> [n_subj, n_actions]
    p_low = softmax(beta * Q_low_sub, axis=1)
    action = np.array([np.random.choice(a=n_actions, p=p_low_subj) for p_low_subj in p_low])
    reward, correct = task.produce_reward(action)

    # Forget Q-values a little bit
    Q_low = (1 - forget) * Q_low + forget * alien_initial_Q  # Q_low.shape -> [n_subj, n_TS, n_aliens, n_actions]
    Q_high = (1 - forget_high) * Q_high + forget_high * alien_initial_Q

    # Calculate RPEs & update Q-values
    current_trial_high = np.arange(n_subj), season, TS
    RPE_high = reward - Q_high[current_trial_high]
    Q_high[current_trial_high] += alpha_high * RPE_high

    current_trial_low = np.arange(n_subj), TS, alien, action
    RPE_low = reward - Q_low[current_trial_low]
    Q_low[current_trial_low] += alpha * RPE_low

    if verbose:
        print("Q_high_sub:\n", Q_high_sub.round(3))
        print("p_high:\n", p_high.round(3))
        print("TS:", TS)
        print("Q_low_sub:\n", Q_low_sub.round(3))
        print("p_low:\n", p_low.round(3))
        print("action:", action)
        print("reward:", reward)
        print("correct:", correct)
        print("RPE_low:", RPE_low.round(3))
        print("RPE_high:", RPE_high.round(3))
        print("new Q_high_sub:\n", Q_high[np.arange(n_subj), season].round(3))
        print("new Q_low_sub:\n", Q_low[np.arange(n_subj), TS, alien].round(3))

    return [Q_low, Q_high, TS, action, correct, reward, p_low]


def update_Qs(season, alien, action, reward,
              Q_low, Q_high,
              beta, beta_high, alpha, alpha_high, forget, forget_high, n_subj, n_TS):

    # Select TS
    Q_high_sub = Q_high[T.arange(n_subj), season]  # Q_high_sub.shape -> [n_subj, n_TS]
    # TS = season  # Flat
    # TS = T.argmax(Q_high_sub, axis=1)  # Hierarchical deterministic
    p_high = T.nnet.softmax(beta_high * Q_high_sub)
    rand = rs.uniform(size=(n_subj, 1))
    cumsum = T.extra_ops.cumsum(p_high, axis=1)
    TS = n_TS - T.sum(rand < cumsum, axis=1)

    # Calculate action probabilities based on TS
    Q_low_sub = Q_low[T.arange(n_subj), TS, alien]  # Q_low_sub.shape -> [n_subj, n_actions]
    p_low = T.nnet.softmax(beta * Q_low_sub)

    # Forget Q-values a little bit
    Q_low = (1 - forget) * Q_low + forget * alien_initial_Q
    Q_high = (1 - forget_high) * Q_high + forget_high * alien_initial_Q

    # Calculate RPEs & update Q-values
    current_trial_high = T.arange(n_subj), season, TS
    RPE_high = reward - Q_high[current_trial_high]
    Q_high = T.set_subtensor(Q_high[current_trial_high],
                             Q_high[current_trial_high] + alpha_high * RPE_high)

    current_trial_low = T.arange(n_subj), TS, alien, action
    RPE_low = reward - Q_low[current_trial_low]
    Q_low = T.set_subtensor(Q_low[current_trial_low],
                            Q_low[current_trial_low] + alpha * RPE_low)

    return [Q_low, Q_high, TS, p_low]


def split_subj_in_half(n_subj):

    half_of_subj = np.arange(0, n_subj, 2)  # np.random.choice(range(n_subj), size=int(np.ceil(n_subj / 2)), replace=False)
    other_half = [i for i in range(n_subj) if i not in half_of_subj]
    half_of_subj = half_of_subj[:len(other_half)]

    return half_of_subj, other_half


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
