verbose = False
import numpy as np
import theano.tensor as T
import theano

from theano.tensor.shared_randomstreams import RandomStreams
rs = RandomStreams()


# Initial Q-value for actions and TS
alien_initial_Q = 1.2  # rewards are z-scored!

# Function to update Q-values based on stimulus, action, and reward
def update_Qs(season, alien, action, reward, Q_low, Q_high, beta_high, alpha, alpha_high, forget, forget_high, n_subj):

    # Loop over trials: take data for all subjects, 1 single trial

    # Select TS
    Q_high_sub = Q_high[T.arange(n_subj), season]
    p_high = T.nnet.softmax(beta_high * Q_high_sub)
    T.printing.Print('Q_high_sub')(Q_high_sub)
    T.printing.Print('p_high')(p_high)
    # TS = season  # Flat
    TS = p_high.argmax(axis=1)  # Hierarchical deterministic

    # def twodchoice(p, a):
    #     return np.random.choice(a=a, p=p / T.sum(p))
    #
    # TS, _ = theano.scan(fn=twodchoice,  # TypeError: shape must be a vector or list of scalar, got 'TensorConstant{1}'
    #                     sequences=p_high,
    #                     non_sequences=3)
    # TS = rs.choice(a=3, size=[31], p=p_high)  # choice(self, size=(), a=2, replace=True, p=None, ndim=None, dtype='int64')

    T.printing.Print('TS')(TS)
    T.printing.Print('alien')(alien)
    T.printing.Print('action')(action)

    # Forget Q-values a little bit
    Q_low_new = (1 - forget) * Q_low + forget * alien_initial_Q
    Q_high_new = (1 - forget_high) * Q_high + forget_high * alien_initial_Q

    # Calculate RPEs & update Q-values
    RPE_low = reward - Q_low_new[T.arange(n_subj), TS, alien, action]
    T.printing.Print('Q_low_old')(Q_low_new[T.arange(n_subj), TS, alien, action])
    Q_low_new = T.set_subtensor(Q_low_new[T.arange(n_subj), TS, alien, action],
                                Q_low_new[T.arange(n_subj), TS, alien, action] + alpha * RPE_low)
    T.printing.Print('reward')(reward)
    T.printing.Print('RPE_low')(RPE_low)
    T.printing.Print('Q_low_new')(Q_low_new[T.arange(n_subj), TS, alien, action])

    RPE_high = reward - Q_high_new[T.arange(n_subj), season, TS]
    Q_high_new = T.set_subtensor(Q_high_new[T.arange(n_subj), season, TS],
                                 Q_high_new[T.arange(n_subj), season, TS] + alpha_high * RPE_high)

    return [Q_low_new, Q_high_new]

# Same, but without theano
def update_Qs_sim(season, TS, alien, action, reward, Q_low, Q_high, alpha, alpha_high, beta_high, forget, forget_high, n_subj, verbose=False):
    # Loop over trials: take data for all subjects, 1 single trial

    # Forget Q-values a little bit
    Q_low = (1 - forget) * Q_low + forget * alien_initial_Q
    Q_high = (1 - forget_high) * Q_high + forget_high * alien_initial_Q

    # Calculate RPEs & update Q-values
    current_trial_high = np.arange(n_subj), season, TS
    RPE_high = reward - Q_high[current_trial_high]
    Q_high[current_trial_high] += alpha_high * RPE_high

    current_trial_low = np.arange(n_subj), TS, alien, action
    RPE_low = reward - Q_low[current_trial_low]
    Q_low[current_trial_low] += alpha * RPE_low

    if verbose:
        print("TS:", TS)
        print("RPE_low:", RPE_low)
        print("RPE_high:", RPE_high)

    return [Q_low, Q_high]

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