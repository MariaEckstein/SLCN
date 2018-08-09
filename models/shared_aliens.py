alien_initial_Q = 5 / 3
verbose = False
import numpy as np
import theano.tensor as T


# Define function to update Q-values based on stimulus, action, and reward
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

# Define function to update Q-values based on stimulus, action, and reward
def update_Q_low(season, alien, action, reward, Q_low, alpha, forget, n_subj):
    # Loop over trials: take data for all subjects, 1 single trial
    Q_low_new = (1 - forget) * Q_low
    RPE_low = alpha * (reward - Q_low_new[T.arange(n_subj), season, alien, action])
    Q_low_new = T.set_subtensor(Q_low_new[T.arange(n_subj), season, alien, action],
                                Q_low_new[T.arange(n_subj), season, alien, action] + RPE_low)

    if verbose:
        T.printing.Print('Q_low.shape')(Q_low.shape)
        T.printing.Print('forget.shape')(forget.shape)
        T.printing.Print('Q_low inside update_Q_low')(Q_low_new)
        T.printing.Print('alien inside update_Q_low')(alien)
        T.printing.Print('action inside update_Q_low')(action)
        T.printing.Print('reward inside update_Q_low')(reward)
        T.printing.Print('RPE_low')(RPE_low)

    return Q_low_new
