from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np
import time

from modeling_helpers_bas import sample_TS_series
from shared_aliens import alien_initial_Q, get_alien_paths, read_in_human_data


# Define things
n_actions, n_aliens, n_seasons, n_TS = 3, 4, 3, 3
human_data_path = get_alien_paths()["human data prepr"]

alpha, beta, forget, alpha_high, beta_high, forget_high = 0.1, 5, 0.001, 0.1, 5, 0.001
subj_id = 0  # which subject should be fitted?
n_samples_MCMC = 10000
beta_MCMC = 1/1e3  # 1/1e3 -> acceptance rate ~ 50%; 1/1e4 -> acceptance rate ~ 20%
verbose = False

correct_TS = np.array([[[1, 6, 1],  # alien0, items0-2
                        [1, 1, 4],  # alien1, items0-2
                        [5, 1, 1],  # etc.
                        [10, 1, 1]],
                       # TS1
                       [[1, 1, 2],  # alien0, items0-2
                        [1, 8, 1],  # etc.
                        [1, 1, 7],
                        [1, 3, 1]],
                       # TS2
                       [[1, 1, 7],  # TS2
                        [3, 1, 1],
                        [1, 3, 1],
                        [2, 1, 1]]])

# Get human data
start = time.time()
n_subj, alien_series, season_series, hum_corrects, action_series, reward_series, _, _ = read_in_human_data(human_data_path, 828, n_aliens, n_actions)

# Replace missing trials with random actions
missing_trials = action_series < 0
action_series[missing_trials] = np.random.randint(0, n_actions, np.sum(missing_trials))
reward_series[missing_trials] = correct_TS[season_series[missing_trials], alien_series[missing_trials], action_series[missing_trials]]
end = time.time()
print("Reading in human data took {} seconds.".format(end - start))

# Run MCMC to get TS
start = time.time()
TS_series_samples, L_samples, accepted_samples = \
    sample_TS_series(season_series[:, subj_id], alien_series[:, subj_id], action_series[:, subj_id], reward_series[:, subj_id],
                     alpha, beta, forget, alpha_high, beta_high, forget_high,
                     n_TS, n_aliens, n_actions, n_samples_MCMC, beta_MCMC, verbose=verbose)
end = time.time()
print("Running TS_series_samples with {0} samples took {1} seconds.".format(n_samples_MCMC, end - start))

# Look at best sample
best_sample = TS_series_samples[np.argwhere(L_samples == np.max(L_samples)).flatten()[0]]
r_best_sample = pearsonr(best_sample, season_series[:, 0])
r_all = [pearsonr(sample, season_series[:, 0])[0] for sample in TS_series_samples]
p_all = [pearsonr(sample, season_series[:, 0])[1] for sample in TS_series_samples]

# Verify that MH is working
colors = ['green' if accepted_samples[i] else 'red' for i in range(len(accepted_samples))]
fig, axes = plt.subplots(nrows=2, ncols=2)
axes[0, 0].set_title("Acceptance rate: {}".format(np.mean(accepted_samples).round(3)))
for i, (L, color, accepted) in enumerate(zip(L_samples, colors, accepted_samples)):
    axes[0, 0].plot(i, L, '.', color=color, label=accepted)
axes[0, 0].set_xlabel("Sample")
axes[0, 0].set_ylabel("Log prob")
# axes[0, 0].legend()
axes[0, 1].set_title("Correlation true and recovered TS")

colors = ['green' if p_all[i] < 0.05 else 'red' for i in range(len(p_all))]
for i, r in enumerate(r_all):
    axes[0, 1].plot(i, r, '.', color=colors[i])
plt.tight_layout()
plt.show()

stop = 42