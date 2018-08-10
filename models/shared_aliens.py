verbose = False
import numpy as np
import theano.tensor as T


alien_initial_Q = 5 / 3

# Update Q-values based on stimulus, action, and reward
def update_Q_low_sim(season, alien, action, reward, Q_low, alpha, forget, n_subj):
    # Loop over trials: take data for all subjects, 1 single trial
    Q_low_new = (1 - forget) * Q_low
    RPE_low = alpha * (reward - Q_low_new[np.arange(n_subj), season, alien, action])
    # Q_low_new[np.arange(n_subj), season, alien, action] = Q_low_new[
    #                                                           np.arange(n_subj), season, alien, action] + RPE_low
    Q_low_new[np.arange(n_subj), season, alien, action] += RPE_low

    if verbose:
        print(Q_low.shape)
        print(forget.shape)
        print(Q_low_new)
        print(alien)
        print(action)
        print(reward)
        print(RPE_low)

    return Q_low_new

# # Update Q-values based on stimulus, action, and reward
# def update_Q_low(season, alien, action, reward, Q_low, alpha, forget, n_subj):
#     # Loop over trials: take data for all subjects, 1 single trial
#     Q_low_new = (1 - forget) * Q_low
#     RPE_low = alpha * (reward - Q_low_new[T.arange(n_subj), season, alien, action])
#     Q_low_new = T.set_subtensor(Q_low_new[T.arange(n_subj), season, alien, action],
#                                 Q_low_new[T.arange(n_subj), season, alien, action] + RPE_low)
#
#     return Q_low_new

# Update Q-values based on stimulus, action, and reward
def update_Q_low(season, alien, action, reward, Q_low, Q_high, alpha_low, alpha_high, beta_high, forget_low, forget_high, n_subj):
    # Loop over trials: take data for all subjects, 1 single trial
    Q_low_new = (1 - forget_low) * Q_low
    RPE_low = alpha_low * (reward - Q_low_new[T.arange(n_subj), season, alien, action])
    Q_low_new = T.set_subtensor(Q_low_new[T.arange(n_subj), season, alien, action],
                                Q_low_new[T.arange(n_subj), season, alien, action] + RPE_low)

    return [Q_low_new, Q_high]


# Update Q-values based on stimulus, action, and reward
def update_Qs(season, alien, action, reward, Q_low, Q_high, alpha_low, alpha_high, beta_high, forget_low, forget_high, n_subj):
    # Loop over trials: take data for all subjects, 1 single trial

    # Select TS
    Q_high_sub = Q_high[T.arange(n_subj), season]
    p_high = T.exp(beta_high * Q_high_sub)
    TS = p_high.argmax(axis=1)  # Select TS deterministically
    # TS = pm.Categorical('TS', p=p_high, shape=n_subj, testval=np.ones(n_subj), observed=np.random.choice(n_subj))  # Select probabilistically

    # Forget Q-values a little bit
    Q_low_new = (1 - forget_low) * Q_low
    Q_high_new = (1 - forget_high) * Q_high

    # Calculate RPEs & update Q-values
    RPE_low = alpha_low * (reward - Q_low_new[T.arange(n_subj), TS, alien, action])
    Q_low_new = T.set_subtensor(Q_low_new[T.arange(n_subj), TS, alien, action],
                                Q_low_new[T.arange(n_subj), TS, alien, action] + RPE_low)

    RPE_high = alpha_high * (reward - Q_high_new[T.arange(n_subj), season, TS])
    Q_high_new = T.set_subtensor(Q_high_new[T.arange(n_subj), season, TS],
                                 Q_high_new[T.arange(n_subj), season, TS] + RPE_high)

    return [Q_low_new, Q_high_new]
