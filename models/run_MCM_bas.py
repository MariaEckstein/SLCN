import matplotlib.pyplot as plt
import numpy as np
import time
import importlib

import modeling_helpers_bas
importlib.reload(modeling_helpers_bas)
from modeling_helpers_bas import sample_TS_series

import shared_aliens_bas
importlib.reload(shared_aliens_bas)
from shared_aliens_bas import alien_initial_Q, get_alien_paths, read_in_human_data


# Define things
n_actions, n_aliens, n_seasons, n_TS = 3, 4, 3, 3
human_data_path = get_alien_paths(False, True)["human data prepr"]

subj_id = 0  # which subject should be fitted?
correct_TS = np.array([[[1, 6, 1],                        [1, 1, 4],                        [5, 1, 1],                        [10, 1, 1]],                        [[1, 1, 2],[1, 8, 1],[1, 1, 7],[1, 3, 1]],                       [[1, 1, 7],                        [3, 1, 1],                        [1, 3, 1],                        [2, 1, 1]]])

# Get human data
start = time.time()
n_subj, alien_series, season_series, hum_corrects, action_series, reward_series, _, _ = read_in_human_data(human_data_path, 828, n_aliens, n_actions)

# Replace missing trials with random actions
missing_trials = action_series < 0
action_series[missing_trials] = np.random.randint(0, n_actions, np.sum(missing_trials))
reward_series[missing_trials] = correct_TS[season_series[missing_trials], alien_series[missing_trials], action_series[missing_trials]]
end = time.time()
print("Reading in human data took {} seconds.".format(end - start))

alpha, beta, forget, alpha_high, beta_high, forget_high = 0.1, 5, 0.001, 0.1, 5, 0.001
n_samples_MCMC = 1000
beta_MCMC = 1  # 1/1e3 -> acceptance rate ~ 50%; 1/1e4 -> acceptance rate ~ 20%
verbose = False

def test_MCMC():
    start = time.time()
    TS_series_samples, L_samples, accepted_samples = sample_TS_series(season_series[:, subj_id], alien_series[:, subj_id], action_series[:, subj_id], reward_series[:, subj_id],
                         alpha, beta, forget, alpha_high, beta_high, forget_high,
                         n_TS, n_aliens, n_actions, n_samples_MCMC, beta_MCMC, verbose=verbose)
    end = time.time()
    print("Running TS_series_samples with {0} samples took {1} seconds.".format(n_samples_MCMC, end - start))
    # Look at best sample
    best_sample = TS_series_samples[np.argwhere(L_samples == np.max(L_samples)).flatten()[0]]
    # Verify that MH is working
    colors = ['green' if accepted_samples[i] else 'red' for i in range(len(accepted_samples))]
    fig,axes = plt.subplots()
    axes.set_title("Acceptance rate: {}".format(np.mean(accepted_samples).round(3)))
    for i, (L, color, accepted) in enumerate(zip(L_samples, colors, accepted_samples)):
        axes.plot(i, L, '.', color=color, label=accepted)
    axes.set_xlabel("Sample")
    axes.set_ylabel("Log prob")
    plt.show()
    return TS_series_samples, L_samples, accepted_samples

TS_series_samples, L_samples, accepted_samples = test_MCMC()

# Run MCMC to get TS

#fig, axes = plt.subplots(nrows=2, ncols=2)
#axes[0, 0].set_title("Acceptance rate: {}".format(np.mean(accepted_samples).round(3)))
#for i, (L, color, accepted) in enumerate(zip(L_samples, colors, accepted_samples)):
#    axes[0, 0].plot(i, L, '.', color=color, label=accepted)
#axes[0, 0].set_xlabel("Sample")
#axes[0, 0].set_ylabel("Log prob")
# axes[0, 0].legend()
#plt.tight_layout()
#plt.show()

#stop = 42

print(modeling_helpers_bas.compute_logprob(TS_series_samples[0,:], season_series, alien_series, action_series, reward_series,alpha, beta, forget, alpha_high, beta_high, forget_high,n_TS, n_aliens, n_actions))
