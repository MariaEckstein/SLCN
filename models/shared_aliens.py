verbose = False
import numpy as np
import theano.tensor as T
import theano

from theano.tensor.shared_randomstreams import RandomStreams
rs = RandomStreams()


# Initial Q-value for actions and TS
alien_initial_Q = 1.2

# Function to update Q-values based on stimulus, action, and reward
def update_Qs(season, alien, action, reward, Q_low, Q_high, alpha, alpha_high, beta_high, forget, forget_high, n_subj):
    # Loop over trials: take data for all subjects, 1 single trial

    # Select TS
    Q_high_sub = Q_high[T.arange(n_subj), season]
    p_high = T.exp(beta_high * Q_high_sub)
    T.printing.Print('p_high')(p_high)
    # TS = season  # Flat model
    # TS = p_high.argmax(axis=1)  # Select TS deterministically
    # TS = pm.Categorical('TS', p=p_high, shape=n_subj, testval=np.ones(n_subj), observed=np.random.choice(n_subj))  # Select probabilistically

    TS, _ = theano.scan(fn=lambda TS : rs.choice(a=3, p=p_high, size=T.as_tensor_variable(1)),  # TypeError: shape must be a vector or list of scalar, got 'TensorConstant{1}'
                        sequences=p_high)

    # TS = rs.choice(a=3, size=[31], p=p_high)  # choice(self, size=(), a=2, replace=True, p=None, ndim=None, dtype='int64')
    T.printing.Print('TS')(TS)

    # Forget Q-values a little bit
    Q_low_new = (1 - forget) * Q_low + forget * alien_initial_Q * T.ones_like(Q_low)
    Q_high_new = (1 - forget_high) * Q_high + forget_high * alien_initial_Q * T.ones_like(Q_high)

    # Calculate RPEs & update Q-values
    RPE_low = reward - Q_low_new[T.arange(n_subj), TS, alien, action]
    Q_low_new = T.set_subtensor(Q_low_new[T.arange(n_subj), TS, alien, action],
                                Q_low_new[T.arange(n_subj), TS, alien, action] + alpha * RPE_low)

    RPE_high = reward - Q_high_new[T.arange(n_subj), season, TS]
    Q_high_new = T.set_subtensor(Q_high_new[T.arange(n_subj), season, TS],
                                 Q_high_new[T.arange(n_subj), season, TS] + alpha_high * RPE_high)

    return [Q_low_new, Q_high_new]

# Same, but without theano
def update_Qs_sim(season, alien, action, reward, Q_low, Q_high, alpha, alpha_high, beta_high, forget, forget_high, n_subj, verbose=False):
    # Loop over trials: take data for all subjects, 1 single trial

    # Select TS
    Q_high_sub = Q_high[np.arange(n_subj), season]
    p_high = np.exp(beta_high * Q_high_sub)
    # TS = season  # Flat model
    # TS = p_high.argmax(axis=1)  # Select TS deterministically
    # TS = pm.Categorical('TS', p=p_high, shape=n_subj, testval=np.ones(n_subj), observed=np.random.choice(n_subj))  # Select probabilistically
    TS = np.array([np.random.choice(a=3, p=p_high_subj / np.sum(p_high_subj)) for p_high_subj in p_high])  # numpy.random.choice(a, size=None, replace=True, p=None)¶

    # Forget Q-values a little bit
    Q_low_new = (1 - forget) * Q_low + forget * alien_initial_Q * np.ones(Q_low.shape)
    Q_high_new = (1 - forget_high) * Q_high + forget_high * alien_initial_Q * np.ones(Q_high.shape)

    # Calculate RPEs & update Q-values
    RPE_low = reward - Q_low_new[np.arange(n_subj), TS, alien, action]
    Q_low_new[np.arange(n_subj), TS, alien, action] += alpha * RPE_low

    RPE_high = reward - Q_high_new[np.arange(n_subj), season, TS]
    Q_high_new[np.arange(n_subj), season, TS] += alpha_high * RPE_high

    if verbose:
        print("TS:", TS)
        print("RPE_low:", RPE_low)
        print("RPE_high:", RPE_high)

    return [Q_low_new, Q_high_new]
