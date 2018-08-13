verbose = False
import numpy as np
import theano.tensor as T


alien_initial_Q = 1.2

# Update Q-values based on stimulus, action, and reward
def update_Q_low_sim(season, alien, action, reward, Q_low, Q_high, alpha, alpha_high, beta_high, forget, forget_high, n_subj, verbose=False):
    # Loop over trials: take data for all subjects, 1 single trial
    Q_low_new = (1 - forget) * Q_low
    RPE_low = alpha * (reward - Q_low_new[np.arange(n_subj), season, alien, action])
    Q_low_new[np.arange(n_subj), season, alien, action] += RPE_low

    return [Q_low_new, Q_high]


# Update Q-values based on stimulus, action, and reward
def update_Q_low(season, alien, action, reward, Q_low, Q_high, alpha, alpha_high, beta_high, forget, forget_high, n_subj):
    # Loop over trials: take data for all subjects, 1 single trial
    Q_low_new = (1 - forget) * Q_low
    RPE_low = alpha * (reward - Q_low_new[T.arange(n_subj), season, alien, action])
    Q_low_new = T.set_subtensor(Q_low_new[T.arange(n_subj), season, alien, action],
                                Q_low_new[T.arange(n_subj), season, alien, action] + RPE_low)

    return [Q_low_new, Q_high]


# Update Q-values based on stimulus, action, and reward
def update_Qs(season, alien, action, reward, Q_low, Q_high, alpha, alpha_high, beta_high, forget, forget_high, n_subj):
    # Loop over trials: take data for all subjects, 1 single trial

    # Select TS
    Q_high_sub = Q_high[T.arange(n_subj), season]
    p_high = T.exp(beta_high * Q_high_sub)
    TS = p_high.argmax(axis=1)  # Select TS deterministically
    # TS = pm.Categorical('TS', p=p_high, shape=n_subj, testval=np.ones(n_subj), observed=np.random.choice(n_subj))  # Select probabilistically

    # Forget Q-values a little bit
    Q_low_new = (1 - forget) * Q_low + forget * alien_initial_Q * T.ones_like(Q_low)
    Q_high_new = (1 - forget_high) * Q_high + forget_high * alien_initial_Q * T.ones_like(Q_high)
    # Q_high_new = Q_high.copy()  # TODO: debugging only

    # Calculate RPEs & update Q-values
    RPE_low = (reward - Q_low_new[T.arange(n_subj), TS, alien, action])
    Q_low_new = T.set_subtensor(Q_low_new[T.arange(n_subj), TS, alien, action],
                                Q_low_new[T.arange(n_subj), TS, alien, action] + alpha * RPE_low)

    RPE_high = (reward - Q_high_new[T.arange(n_subj), season, TS])
    Q_high_new = T.set_subtensor(Q_high_new[T.arange(n_subj), season, TS],
                                 Q_high_new[T.arange(n_subj), season, TS] + alpha_high * RPE_high)

    return [Q_low_new, Q_high_new]

# Update Q-values based on stimulus, action, and reward
def update_Qs_sim(season, alien, action, reward, Q_low, Q_high, alpha, alpha_high, beta_high, forget, forget_high, n_subj, verbose=False):
    # Loop over trials: take data for all subjects, 1 single trial

    # Select TS
    Q_high_sub = Q_high[np.arange(n_subj), season]
    p_high = np.exp(beta_high * Q_high_sub)
    TS = p_high.argmax(axis=1)  # Select TS deterministically
    # TS = season  # TODO: this line is just for debugging!
    # TS = pm.Categorical('TS', p=p_high, shape=n_subj, testval=np.ones(n_subj), observed=np.random.choice(n_subj))  # Select probabilistically

    # Forget Q-values a little bit
    Q_low_new = (1 - forget) * Q_low + forget * alien_initial_Q * np.ones(Q_low.shape)
    Q_high_new = (1 - forget_high) * Q_high + forget_high * alien_initial_Q * np.ones(Q_high.shape)

    # Calculate RPEs & update Q-values
    RPE_low = (reward - Q_low_new[np.arange(n_subj), TS, alien, action])
    Q_low_new[np.arange(n_subj), TS, alien, action] += alpha * RPE_low

    RPE_high = (reward - Q_high_new[np.arange(n_subj), season, TS])
    Q_high_new[np.arange(n_subj), season, TS] += alpha_high * RPE_high

    if verbose:
        print("TS:", TS)
        print("RPE_low:", RPE_low)
        print("RPE_high:", RPE_high)

    return [Q_low_new, Q_high_new]
