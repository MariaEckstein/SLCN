import numpy as np
from shared_aliens import alien_initial_Q, softmax


def get_Qhigh_series(TS_series, season_series, reward_series, alpha_high, forget_high, n_TS):
    Q_high_series = []
    Q_high = alien_initial_Q * np.ones((n_TS, n_TS))  # first trial Q_high
    for TS, season, reward in zip(TS_series, season_series, reward_series):  # iterate over trials
        Q_high = (1 - forget_high) * Q_high + forget_high * alien_initial_Q
        current_trial_high = season, TS
        Q_high[current_trial_high] += alpha_high * (reward - Q_high[current_trial_high])
        Q_high_series = Q_high_series + [Q_high]
    return np.array(Q_high_series)


def get_Qlow_series(TS_series, alien_series, action_series, reward_series, alpha, forget, n_TS, n_aliens, n_actions):
    Q_low_series = []
    Q_low = alien_initial_Q * np.ones((n_TS, n_aliens, n_actions))
    for TS, alien, action, reward in zip(TS_series, alien_series, action_series, reward_series):
        Q_low = (1 - forget) * Q_low + forget * alien_initial_Q
        current_trial_low = TS, alien, action
        Q_low[current_trial_low] += alpha * (reward - Q_low[current_trial_low])
        Q_low_series = Q_low_series + [Q_low]
    return np.array(Q_low_series)


def get_phigh_series(Q_high_series, season_series, beta_high):
    return np.array([softmax(beta_high * Q_high[season]) for Q_high, season in zip(Q_high_series, season_series)])


def get_plow_series(Q_low_series, TS_series, alien_series, beta):
    return np.array([softmax(beta * Q_low[TS, alien]) for Q_low, TS, alien in zip(Q_low_series, TS_series, alien_series)])


def get_initial_TS_series(season_series):
    return season_series  # How about this? It would basically initialize it at the flat RL agent.


def generate_TS_series(TS_series, n_TS):
    #this replaces a random entry in TS_series by a random alternative TS
    #which serves a proposal distribution for the MCMC chain that we're building
    # TS_series_new = TS_series.copy()
    TS_series_new = np.random.randint(0, n_TS, len(TS_series))
    return TS_series_new


def compute_logprob(TS_series, season_series, alien_series, action_series, reward_series,
                    alpha, beta, forget, alpha_high, beta_high, forget_high,
                    n_TS, n_aliens, n_actions):
    Q_high_series =  get_Qhigh_series(TS_series, season_series, reward_series, alpha_high, forget_high, n_TS)
    Q_low_series = get_Qlow_series(TS_series, alien_series, action_series, reward_series, alpha, forget, n_TS, n_aliens, n_actions)
    phigh_series = get_phigh_series(Q_high_series, season_series, beta_high)
    plow_series = get_plow_series(Q_low_series, TS_series, alien_series, beta)

    return np.sum(np.log(plow_series)) + np.sum(np.log(phigh_series))


def accept(beta, delta_L):
    # Metropolis-Hastings acceptance rule:
    # - always accept the new logprob when it is better than the old one
    # - also accept the new logprob with higher probabilities the smaller the (negative) difference to the old one
    return delta_L > 0 or np.random.uniform() > np.exp(beta*delta_L)


def sample_TS_series(season_series, alien_series, action_series, reward_series,
                     alpha, beta, forget, alpha_high, beta_high, forget_high,
                     n_TS, n_aliens, n_actions, n_samples_MCMC, beta_MCMC, verbose=False):
    # Sample TS series for a single subject

    TS_series_samples = []
    L_samples = []
    accepted_samples = []

    TS_series = get_initial_TS_series(season_series)
    L = compute_logprob(TS_series, season_series, alien_series, action_series, reward_series,
                        alpha, beta, forget, alpha_high, beta_high, forget_high,
                        n_TS, n_aliens, n_actions)

    for i in range(n_samples_MCMC):
        if verbose:
            print("Drawing MCMC sample {}.".format(i))
        TS_series_new = generate_TS_series(TS_series, n_TS)
        L_new = compute_logprob(TS_series_new, season_series, alien_series, action_series, reward_series,
                                alpha, beta, forget, alpha_high, beta_high, forget_high,
                                n_TS, n_aliens, n_actions)
        accept_sample = accept(beta_MCMC, L_new - L)
        if accept_sample:
            if verbose:
                print("Accepted!")
            TS_series = TS_series_new.copy()  # Without copy, TS_series would be overwritten next time we set TS_series_new
            L = L_new.copy()  # same
        TS_series_samples += [TS_series]
        L_samples += [L]
        accepted_samples += [accept_sample]

    return np.array(TS_series_samples), np.array(L_samples), np.array(accepted_samples)