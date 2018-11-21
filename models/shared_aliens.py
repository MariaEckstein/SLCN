from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import theano.tensor as T

from theano.tensor.shared_randomstreams import RandomStreams
rs = RandomStreams()


# Initial Q-value for actions and TS
alien_initial_Q = 1.2


def get_alien_paths(run_on_cluster=False):

    if run_on_cluster:
        base_path = '/home/bunge/maria/Desktop/'
        return {'human data': base_path + '/AlienshumanData/',
                'human data prepr': base_path + '/AlienshumanData/prepr/',
                'fitting results': base_path + '/AliensPyMC3/fitting/',
                'simulations': base_path + 'AliensPyMC3/Aliensimulations/'}

    else:
        base_path = 'C:/Users/maria/MEGAsync/Berkeley/TaskSets'
        return {'human data': base_path + '/Data/versions1.0and3.1/',
                'human data prepr': base_path + '/Data/version3.1preprocessed/',  # '/Data/versions1.0and3.1preprocessed',
                'fitting results': base_path + '/AliensFitting/',
                'simulations': 'C:/Users/maria/MEGAsync/SLCN/PSsimulations/'}


# Same, but without theano, and selecting actions rather than reading them in from a file
def update_Qs_sim(season, alien,
                  Q_low, Q_high,
                  beta, beta_high, alpha, alpha_high, forget, forget_high,
                  n_subj, n_actions, n_TS, task, verbose=False):

    # Select TS
    # Q_high_sub = Q_high[np.arange(n_subj), season]  # Q_high_sub.shape -> (n_subj, n_TS)
    # p_high = softmax(beta_high * Q_high_sub, axis=1)   # p_high.shape -> (n_subj, n_TS)
    TS = season  # Flat
    # TS = 0  # fs
    # TS = Q_high_sub.argmax(axis=1)  # Hierarchical deterministic
    # TS = np.array([np.random.choice(a=n_TS, p=p_high_subj) for p_high_subj in p_high])  # Hierarchical softmax

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


def simulate_competition_phase(model_name, final_Q_high, final_Q_low, task,
                               n_seasons, n_aliens, n_sim, beta_high):
    n_blocks_comp = 3
    comp_data = pd.DataFrame(np.full((1, 4), np.nan),
                             columns=['perc_selected_better', 'se', 'choice', 'phase'])

    # Select between two seasons
    if model_name == 'hier':
        season_values = np.max(final_Q_high, axis=2)  # n_sim x n_seasons (value of highest-valued TS for each season)
    elif model_name == 'flat':
        best_alien_values = np.max(final_Q_low, axis=3)  # n_sim x n_seasons x n_aliens (alien values if correct action)
        season_values = np.mean(best_alien_values, axis=2)  # n_sim x n_seasons (average over aliens in each season)

    for i, season_pair in enumerate(combinations(range(n_seasons), 2)):

        # Let agents choose (using softmax)
        season_probs = softmax(beta_high * season_values[:, season_pair], axis=1)  # prob. of selecting each season in the pair
        season_choice = np.array([np.random.choice(a=season_pair, size=n_blocks_comp, p=sim_p) for sim_p in season_probs])

        # Calculate stats
        selected_better_sim = np.mean(season_choice == min(season_pair), axis=1)
        selected_better_mean, selected_better_se = np.mean(selected_better_sim), np.std(selected_better_sim) / np.sqrt(len(selected_better_sim))
        comp_data.loc[i] = [selected_better_mean, selected_better_se, str(season_pair), 'season']

    # Select between two aliens in the same season
    p_TS = softmax(beta_high.reshape(n_sim, 1, 1) * final_Q_high, axis=2)  # n_sim x n_seasons x n_TS (prob. of each TS in each season)
    alien_value = np.max(final_Q_low, axis=3)  # n_sim x n_seasons x n_aliens (alien values for correct action)
    for season in range(n_seasons):

        # Select TS for given season
        if model_name == 'hier':
            selected_TS = [np.random.choice(a=range(n_seasons), p=p_sim) for p_sim in p_TS[:, season]]  # n_sim (TS selected by each sim)
        elif model_name == 'flat':
            selected_TS = season * np.ones(n_sim, dtype=int)

        # Select alien given TS
        for alien_pair in combinations(range(n_aliens), 2):
            alien_values = np.array([alien_value[np.arange(n_sim), selected_TS, alien] for alien in alien_pair])
            alien_probs = softmax(alien_values, axis=0)
            alien_choice = np.array([np.random.choice(a=alien_pair, size=n_blocks_comp, p=sim_p) for sim_p in alien_probs.T])

            # Calculate stats
            true_alien_values = np.max(task.TS[season], axis=1)[list(alien_pair)]
            better_alien = alien_pair[np.argmax(true_alien_values)]
            selected_better_sim = np.mean(alien_choice == better_alien, axis=1)
            selected_better_mean, selected_better_se = np.mean(selected_better_sim), np.std(selected_better_sim) / np.sqrt(len(selected_better_sim))
            comp_data.loc[i+1] = [selected_better_mean, selected_better_se, str(season) + str(alien_pair), 'season-alien']
            i += 1

    return comp_data


def simulate_rainbow_phase(n_seasons, n_aliens, n_actions, n_TS,
                           model_name, n_sim,
                           beta, beta_high, final_Q_low, final_Q_high):

    n_blocks_rainbow = 4  # TODO: Check!
    rainbow_dat = np.full((n_aliens, n_actions), np.nan)

    if model_name == 'hier':
        # Hierarchical agents first select a TS, then an action within this TS, according to the seen alien.
        # TS are selected by comparing their maximum values, i.e., the values in the season in which they are correct
        # Actions are selected normally, i.e., by comparing the values of the different actions, given the alien.
        # TS and actions are selected using softmax (rather than, e.g., max).

        # Select TS
        Q_TS = np.max(final_Q_high, axis=1)  # n_sim x TS
        p_TS = softmax(beta_high * Q_TS, axis=1)  # n_sim x TS
        TS = np.array([np.random.choice(a=range(n_TS), size=n_blocks_rainbow, p=p_sim) for p_sim in p_TS])  # n_sim x n_blocks_rainbow

        # Select action
        for i, alien in enumerate(range(n_aliens)):
            Q_alien_in_this_TS = final_Q_low[np.arange(n_sim), TS[:, alien], alien * np.ones(n_sim, dtype=int)]  # n_sim x n_actions
            p_action = softmax(beta * Q_alien_in_this_TS, axis=1)  # n_sim x n_actions
            choice = np.array([np.random.choice(a=range(n_actions), size=n_blocks_rainbow, p=p_sim) for p_sim in p_action])  # n_sim x n_blocks_rainbow
            rainbow_dat[alien] = [np.sum(choice == a) for a in range(n_actions)]

    elif model_name == 'flat':
        # Flat agents select actions based on how much reward they haven given for this alien, across seasons.

        selected_actions = np.full((n_sim, n_aliens, n_blocks_rainbow), np.nan)
        for sim in range(n_sim):
            for alien in range(n_aliens):
                Q_actions_alien = final_Q_low[sim, :, alien, :]  # season x action
                p_actions_alien = softmax(Q_actions_alien.flatten()).reshape(n_seasons, n_actions)  # season x action (flatten then reshape to get the softmax over all entries, not rowwise or columnwise)
                selected_action = np.random.choice(a=np.tile(range(n_actions), n_seasons), size=n_blocks_rainbow, p=p_actions_alien.flatten())  # n_blocks_rainbow x 1  (flatten again and tile the actions)
                selected_actions[sim, alien] = selected_action

        for i, alien in enumerate(range(n_aliens)):
            rainbow_dat[alien] = [np.sum(selected_actions == a) for a in range(n_actions)]

    return rainbow_dat


def get_summary_rainbow(n_aliens, n_seasons, rainbow_dat, task):

    correct_actions = np.argmax(task.TS, axis=2)  # season x alien
    TS_choices = np.array([rainbow_dat[range(n_aliens), correct_actions[TSi]] for TSi in range(n_seasons)])
    TS_choices = np.vstack([TS_choices, (rainbow_dat[0, 0], 0, 0, rainbow_dat[3, 2])])

    if False:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.title('Rainbow phase')
        ax.bar(np.arange(4) + 0.15, np.sum(TS_choices, axis=1), 0.3)
        ax.set_ylabel('TS chosen (count)')
        ax.set_xticks(range(4))
        ax.set_xticklabels(['TS0', 'TS1', 'TS2', 'noTS'])
        ax.legend()

    return TS_choices


def get_summary_initial_learn(seasons, corrects, aliens, actions,
                              n_seasons, n_sim, trials, task):
    season_changes = np.array([seasons[i, 0] != seasons[i + 1, 0] for i in list(trials['1InitialLearn'])[:-1]])
    season_changes = np.insert(season_changes, 0, False)
    season_presentation = np.cumsum(season_changes)
    repetition = season_presentation // n_seasons
    n_trials_per_rep = np.sum(repetition == 0)
    n_rep_rep = np.sum(season_presentation == 0)

    corrects_rep = corrects[trials['1InitialLearn']].reshape((3, n_trials_per_rep, n_sim))
    learning_curve_rep = np.mean(corrects_rep, axis=2)
    rep_rep = learning_curve_rep.reshape((3, 3, n_rep_rep))
    rep_rep = np.mean(rep_rep, axis=1)

    # Get savings (increase in accuracy from first repetition to last repetition)
    saving_first_trial = rep_rep[-1, 0] - rep_rep[0, 0]  # first trial only
    saving_last_trial = rep_rep[-1, -1] - rep_rep[0, -1]  # last trial only
    saving_av = np.mean(rep_rep[-1] - rep_rep[0])  # average over all 40 trials

    # Get cross-over summaries (accuracy according to current TS, previous TS, and other TS)
    first_alien_new_season = aliens[trials['1InitialLearn']][season_changes][1:]  # remove very first round
    first_action_new_season = actions[trials['1InitialLearn']][season_changes][1:]  # because there is not previous TS
    first_acc_new_season = corrects[trials['1InitialLearn']][season_changes][1:]

    current_TS = seasons[trials['1InitialLearn']][season_changes][1:]
    prev_TS = seasons[trials['1InitialLearn']][season_changes][:-1]
    other_TS = 3 - current_TS - prev_TS

    acc_current_TS = task.TS[current_TS, first_alien_new_season, first_action_new_season] > 1
    assert np.all(first_acc_new_season == acc_current_TS)
    acc_prev_TS = task.TS[prev_TS, first_alien_new_season, first_action_new_season] > 1
    acc_other_TS = task.TS[other_TS, first_alien_new_season, first_action_new_season] > 1

    acc_current_prev_other = [np.mean(acc_current_TS), np.mean(acc_prev_TS), np.mean(acc_other_TS)]

    return [saving_av, saving_first_trial, saving_last_trial] + acc_current_prev_other


def get_summary_cloudy(seasons, corrects, n_seasons, n_sim, trials):
    season_changes = np.array([seasons[i, 0] != seasons[i + 1, 0] for i in list(trials['2CloudySeason'])[:-1]])
    season_changes = np.insert(season_changes, 0, False)
    season_presentation = np.cumsum(season_changes)
    n_rep_rep = np.sum(season_presentation == 0)

    corrects_rep = corrects[trials['2CloudySeason']].reshape((int(corrects[trials['2CloudySeason']].shape[0] / n_rep_rep), n_rep_rep, n_sim))
    learning_curve_rep = np.mean(corrects_rep, axis=2)

    acc_trials_0to3 = np.mean(learning_curve_rep[:, :4], axis=0)

    if False:
        plt.figure()
        for rep in range(9):
            plt.plot(range(40), learning_curve_rep[rep], label=rep)
        plt.legend()
        plt.show()

    return acc_trials_0to3


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
