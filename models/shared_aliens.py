verbose = False
import numpy as np
import theano.tensor as T
import theano
import pymc3 as pm

from theano.tensor.shared_randomstreams import RandomStreams
rs = RandomStreams()


# Initial Q-value for actions and TS
alien_initial_Q = 1  # rewards are z-scored!

def logp(val):
    return T.log(1/3)

# def logp_TS():
#     # most_likely = T.argmax(p_high, axis=1)
#     def logp(val):
#         return T.log(1 - (val - most_likely) ** 2) # 1 - squared error from most likely
#     return logp

class CustomTSDist(pm.DiscreteUniform):
    def __init__(self, lower, upper, p, *args, **kwargs):
        super(CustomTSDist, self).__init__(lower, upper, *args, **kwargs)
        self.p = p

    def logp(self, value):
        upper = self.upper
        lower = self.lower
        return pm.distributions.dist_math.bound(-T.log(upper - lower + 1), lower <= value, value <= upper)

    # def random(self, point=None, size=None, repeat=None):
    #     # Produce sample
    #     pass


# Function to update Q-values based on stimulus, action, and reward
def update_Qs(season, alien, action, reward,
              Q_low, Q_high,
              beta_high, alpha, alpha_high, forget, forget_high, n_subj):

    # Select TS
    Q_high_sub = Q_high[T.arange(n_subj), season]
    p_high = T.nnet.softmax(beta_high * Q_high_sub)
    T.printing.Print('p_high')(p_high)

    TS = CustomTSDist('TS', 0, 2, p_high)
    T.printing.Print('TS')(TS)

    # TS.dimshuffle(0)
    # TS = TS.flatten(ndim=1)

    # Participant selects action based on TS and observes a reward

    # Forget Q-values a little bit
    Q_low = (1 - forget) * Q_low + forget * alien_initial_Q
    Q_high = (1 - forget_high) * Q_high + forget_high * alien_initial_Q  # TODO THIS IS WHERE THE NANS ARE INTRODUCED

    # Calculate RPEs & update Q-values
    current_trial_high = T.arange(n_subj), season, TS
    RPE_high = reward - Q_high[current_trial_high]
    Q_high = T.set_subtensor(Q_high[current_trial_high],
                             Q_high[current_trial_high] + alpha_high * RPE_high)

    current_trial_low = T.arange(n_subj), TS, alien, action
    RPE_low = reward - Q_low[current_trial_low]
    Q_low = T.set_subtensor(Q_low[current_trial_low],
                            Q_low[current_trial_low] + alpha * RPE_low)

    return [Q_low, Q_high, TS]


# Same, but without theano
def update_Qs_sim(season, alien,
                  Q_low, Q_high,
                  beta, beta_high, alpha, alpha_high, forget, forget_high,
                  n_subj, n_actions, task, verbose=False):

    # Select TS
    Q_high_sub = Q_high[np.arange(n_subj), season]
    p_high = softmax(beta_high * Q_high_sub, axis=1)
    TS = season  # Flat
    # TS = p_high.argmax(axis=1)  # Hierarchical deterministic
    # TS = np.array([np.random.choice(a=3, p=p_high_subj) for p_high_subj in p_high])  # Hierarchical softmax

    # Select action based on TS
    Q_low_sub = Q_low[np.arange(n_subj), TS, alien]
    p_low = softmax(beta * Q_low_sub, axis=1)
    action = [np.random.choice(range(n_actions), p=p_low_subj) for p_low_subj in p_low]
    reward, correct = task.produce_reward(action)

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
        print("Q_high_sub:", Q_high_sub.round(3))
        print("p_high:", p_high.round(3))
        print("TS:", TS)
        print("Q_low_sub:", Q_low_sub.round(3))
        print("p_low:", p_low.round(3))
        print("action:", action)
        print("reward:", reward)
        print("RPE_low:", RPE_low.round(3))
        print("RPE_high:", RPE_high.round(3))
        print("new Q_high_sub:", Q_high[np.arange(n_subj), season].round(3))
        print("new Q_low_sub:", Q_low[np.arange(n_subj), TS, alien].round(3))

    return [Q_low, Q_high, TS, action, correct, reward, p_low]

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