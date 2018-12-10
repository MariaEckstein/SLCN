from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import os
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
    Q_high_sub = Q_high[np.arange(n_subj), season]  # Q_high_sub.shape -> (n_subj, n_TS)
    p_high = softmax(beta_high * Q_high_sub, axis=1)   # p_high.shape -> (n_subj, n_TS)
    # TS = season  # Flat
    # TS = 0  # fs
    # TS = Q_high_sub.argmax(axis=1)  # Hierarchical deterministic
    TS = np.array([np.random.choice(a=n_TS, p=p_high_subj) for p_high_subj in p_high])  # Hierarchical softmax

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
    n_blocks_comp = 30  # 3 in humans; 30 to get a better signal
    comp_data = pd.DataFrame(np.full((1, 4), np.nan),
                             columns=['perc_selected_better', 'se', 'choice', 'phase'])

    # Select between two seasons
    if model_name == 'hier':
        Q_season = np.max(final_Q_high, axis=2)  # n_sim x n_seasons (value of highest-valued TS for each season)
    elif model_name == 'flat':
        Q_alien_corr_action = np.max(final_Q_low, axis=3)  # n_sim x n_seasons x n_aliens (alien values if correct action)
        Q_season = np.mean(Q_alien_corr_action, axis=2)  # n_sim x n_seasons (average over aliens in each season)

    for i, season_pair in enumerate(combinations(range(n_seasons), 2)):

        # Let agents choose (using softmax)
        p_season = softmax(beta_high * Q_season[:, season_pair], axis=1)  # prob. of selecting each season in the pair
        season_choice = np.array([np.random.choice(a=season_pair, size=n_blocks_comp, p=sim_p) for sim_p in p_season])

        # Calculate stats
        selected_better_sim = np.mean(season_choice == min(season_pair), axis=1)
        selected_better_mean, selected_better_se = np.mean(selected_better_sim), np.std(selected_better_sim) / np.sqrt(len(selected_better_sim))
        comp_data.loc[i] = [selected_better_mean, selected_better_se, str(season_pair), 'season']

    # Select between two aliens in the same season
    p_TS = softmax(beta_high.reshape(n_sim, 1, 1) * final_Q_high, axis=2)  # n_sim x n_seasons x n_TS (prob. of each TS in each season)
    Q_alien = np.max(final_Q_low, axis=3)  # n_sim x n_seasons x n_aliens (alien values for correct action)
    for season in range(n_seasons):

        # Select TS for given season
        if model_name == 'hier':
            selected_TS = [np.random.choice(a=range(n_seasons), p=p_sim) for p_sim in p_TS[:, season]]  # n_sim (TS selected by each sim)
        elif model_name == 'flat':
            selected_TS = season * np.ones(n_sim, dtype=int)

        # Select alien given TS
        for alien_pair in combinations(range(n_aliens), 2):
            Q_aliens = np.array([Q_alien[np.arange(n_sim), selected_TS, alien] for alien in alien_pair])
            p_aliens = softmax(Q_aliens, axis=0)
            alien_choice = np.array([np.random.choice(a=alien_pair, size=n_blocks_comp, p=sim_p) for sim_p in p_aliens.T])

            # Calculate stats
            true_Q_aliens = np.max(task.TS[season], axis=1)[list(alien_pair)]
            if true_Q_aliens[0] != true_Q_aliens[1]:
                better_alien = alien_pair[np.argmax(true_Q_aliens)]
                selected_better_sim = np.mean(alien_choice == better_alien, axis=1)
                selected_better_mean, selected_better_se = np.mean(selected_better_sim), np.std(selected_better_sim) / np.sqrt(len(selected_better_sim))
                comp_data.loc[i+1] = [selected_better_mean, selected_better_se, str(season) + str(alien_pair), 'season-alien']
                i += 1

    return comp_data


def simulate_rainbow_phase(n_seasons, model_name, n_sim,
                           beta, beta_high, final_Q_low, final_Q_high):

    if model_name == 'hier':
        # Hierarchical agents first select a TS, then an action within this TS, according to the seen alien.
        # TS are selected by comparing their maximum values, i.e., the values in the season in which they are correct
        # Actions are selected like in the initial learning phase, i.e., by comparing the values of the different
        # actions, given the alien. TS and actions are selected using softmax (rather than, e.g., max).

        # Calculate p(TS) <- softmax(Q(TS))
        Q_TS = np.max(final_Q_high, axis=1)  # n_sim x TS. Find the correct TS in each season.
        p_TS = softmax(beta_high * Q_TS, axis=1)  # n_sim x TS

        # Calculate p(action|alien) <- p(TS) * p(action|alien,TS)
        final_p_low = softmax(final_Q_low, axis=3)  # TODO: add beta *
        p_alien_action_TS = p_TS.reshape(n_sim, n_seasons, 1, 1) * final_p_low
        p_alien_action = np.sum(p_alien_action_TS, axis=1)

    elif model_name == 'flat':
        # Flat agents select actions based on how much reward they haven given for this alien, averaged over seasons.

        Q_alien_action = np.mean(final_Q_low, axis=1)  # n_sim x n_aliens x n_actions. Av. value of each action for each alien
        p_alien_action = softmax(Q_alien_action, axis=2)  # n_sim x n_aliens x n_actions. Corresponding probabilities

    # Average over simulations
    rainbow_dat = np.mean(p_alien_action, axis=0)

    return rainbow_dat


def get_summary_rainbow(n_aliens, n_seasons, rainbow_dat, task):

    # Get number of choices for each TS
    correct_actions = np.argmax(task.TS, axis=2)  # season x alien
    TS_choices = np.array([rainbow_dat[range(n_aliens), correct_actions[TSi]] for TSi in range(n_seasons)])
    TS_choices[[1, 2, 0, 2], [0, 0, 3, 3]] = np.nan  # remove actions that are correct in > 1 TS
    none_choices = (rainbow_dat[0, 0], 0, 0, rainbow_dat[3, 2])
    TS_choices = np.vstack([TS_choices, none_choices])
    summary_rainbow = np.nanmean(TS_choices, axis=1)

    # Get slope
    slope = summary_rainbow[2] - summary_rainbow[0]  # linear contrast: -1, 0, 1

    # Visualize TS_choices
    if False:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.title('Rainbow phase')
        ax.bar(np.arange(4), np.sum(TS_choices, axis=1), 0.3)
        ax.set_ylabel('TS chosen (count)')
        ax.set_xticks(range(4))
        ax.set_xticklabels(['TS0', 'TS1', 'TS2', 'noTS'])
        ax.legend()

    return list(summary_rainbow) + [slope]


def get_summary_initial_learn(seasons, corrects, aliens, actions,
                              n_seasons, n_sim, trials, task):

    # Get savings (increase in accuracy from first repetition to last repetition)
    season_changes = np.array([seasons[i, 0] != seasons[i + 1, 0] for i in list(trials['1InitialLearn'])[:-1]])
    season_changes = np.insert(season_changes, 0, False)
    season_presentation = np.cumsum(season_changes)
    repetition = season_presentation // n_seasons
    n_trials_per_rep = np.sum(repetition == 0)
    n_rep_rep = np.sum(season_presentation == 0)

    corrects_rep = corrects.reshape((3, n_trials_per_rep, n_sim))
    learning_curve_rep = np.mean(corrects_rep, axis=2)
    rep_rep = learning_curve_rep.reshape((3, 3, n_rep_rep))
    rep_rep = np.mean(rep_rep, axis=1)

    saving_first_trial = rep_rep[-1, 0] - rep_rep[0, 0]  # first trial only
    saving_last_trial = rep_rep[-1, -1] - rep_rep[0, -1]  # last trial only
    saving_av = np.mean(rep_rep[-1] - rep_rep[0])  # average over all 40 trials

    savings = [saving_av, saving_first_trial, saving_last_trial]

    # Get intrusion errors (accuracy according to current TS, previous TS, and other TS)
    first_alien_new_season = aliens[season_changes][1:]  # remove very first round
    first_action_new_season = actions[season_changes][1:]  # because there is not previous TS
    first_acc_new_season = corrects[season_changes][1:]

    current_TS = seasons[season_changes][1:]
    prev_TS = seasons[season_changes][:-1]
    other_TS = 3 - current_TS - prev_TS

    first_action_new_season[first_action_new_season < 0] = np.random.choice(a=range(3))  # TODO: think about how to deal with missing values!
    acc_current_TS = task.TS[current_TS, first_alien_new_season, first_action_new_season] > 1
    # assert np.all(first_acc_new_season == acc_current_TS)
    acc_prev_TS = task.TS[prev_TS, first_alien_new_season, first_action_new_season] > 1
    acc_other_TS = task.TS[other_TS, first_alien_new_season, first_action_new_season] > 1

    intrusion_errors = [np.mean(acc_current_TS), np.mean(acc_prev_TS), np.mean(acc_other_TS)]

    # Get performance index for each TS (% correct for aliens with same value in different TS)
    Q6_TS0 = (seasons == 0) * (aliens == 0)  # unique alien-action combo
    Q7_TS1 = (seasons == 1) * (aliens == 2)  # unique alien-action combo
    Q7_TS2 = (seasons == 2) * (aliens == 0)  # same as TS1 (value 2)
    Q7 = [np.mean(corrects[Q6_TS0]), np.mean(corrects[Q7_TS1]), np.mean(corrects[Q7_TS2])]

    # Q2_TS1 = (seasons == 1) * (aliens == 0)  # same correct action as in TS2 (value 4)
    # Q2_TS2 = (seasons == 2) * (aliens == 3)  # same correct action as in TS0 (value 10)
    # Q2 = [np.nan, np.mean(corrects[Q2_TS1]), np.mean(corrects[Q2_TS2])]

    Q4_TS0 = (seasons == 0) * (aliens == 1)  # unique alien-action combo
    Q3_TS1 = (seasons == 1) * (aliens == 3)  # unique alien-action combo
    Q3_TS2 = (seasons == 2) * ((aliens == 1) + (aliens == 2))  # unique alien-action combo
    Q3 = [np.mean(corrects[Q4_TS0]), np.mean(corrects[Q3_TS1]), np.mean(corrects[Q3_TS2])]

    TS_perf = np.nanmean([Q7, Q3], axis=0)

    # Get corr of performance over TS
    corr = np.corrcoef(TS_perf, np.arange(3))

    return savings + intrusion_errors + list(TS_perf) + [corr[0, 1]]


def get_summary_cloudy(seasons, corrects, n_sim, trials_cloudy):

    # Get accuracy for trials 0 to 3 (averaged over TS)
    season_changes = np.array([seasons[i, 0] != seasons[i+1, 0] for i in list(trials_cloudy)[:-1]])
    season_changes = np.insert(season_changes, 0, False)
    season_presentation = np.cumsum(season_changes)
    n_rep_rep = np.sum(season_presentation == 0)

    n_first_trials = int(corrects[trials_cloudy].shape[0] / n_rep_rep)
    corrects_rep = corrects[trials_cloudy].reshape((n_first_trials, n_rep_rep, n_sim))
    learning_curve_rep = np.mean(corrects_rep, axis=2)

    acc_first4 = learning_curve_rep[:, :4]
    acc_first4_mean = np.mean(acc_first4, axis=0)

    # Get a slope for each TS
    seasons_rep = seasons[trials_cloudy].reshape((n_first_trials, n_rep_rep, n_sim))
    slope = np.sum(acc_first4_mean * (np.arange(4) - 1.5))
    TS_slopes = np.zeros(3)

    for TS in range(3):
        corrects_rep_TS = corrects_rep.copy()
        corrects_rep_TS[seasons_rep != TS] = np.nan
        learning_curve_rep_TS = np.nanmean(corrects_rep_TS, axis=2)
        acc_first4_mean_TS = np.mean(learning_curve_rep_TS[:, :4], axis=0)
        slope_TS = np.sum(acc_first4_mean_TS * (np.arange(4) - 1.5))
        TS_slopes[TS] = slope_TS

    if False:
        plt.figure()
        for rep in range(9):
            plt.plot(range(40), learning_curve_rep[rep], label=rep)
        plt.legend()
        plt.show()

    return list(acc_first4_mean) + [slope] + list(TS_slopes)


def read_in_human_data(human_data_path, n_trials, n_aliens, n_actions):
    print("Reading in human data from {}!".format(human_data_path))

    file_names = [file_name for file_name in os.listdir(human_data_path) if "pick" not in file_name]
    n_hum = len(file_names)

    hum_seasons = np.zeros([n_trials, n_hum], dtype=int)
    hum_aliens = np.zeros([n_trials, n_hum], dtype=int)
    hum_actions = np.zeros([n_trials, n_hum], dtype=int)
    hum_rewards = np.zeros([n_trials, n_hum])
    hum_corrects = np.zeros([n_trials, n_hum])
    hum_phase = np.zeros([n_trials, n_hum])
    hum_rainbow_dat = np.zeros((n_hum, n_aliens, n_actions))

    for subj, file_name in enumerate(file_names):
        subj_file = pd.read_csv(human_data_path + '/' + file_name, index_col=0).reset_index(drop=True)

        # Get feed-aliens phases
        feed_aliens_file = subj_file[
            (subj_file['phase'] == '1InitialLearning') | (subj_file['phase'] == '2CloudySeason')]
        hum_seasons[:, subj] = feed_aliens_file['TS']
        hum_aliens[:, subj] = feed_aliens_file['sad_alien']
        hum_actions[:, subj] = feed_aliens_file['item_chosen']
        hum_rewards[:, subj] = feed_aliens_file['reward']
        hum_corrects[:, subj] = feed_aliens_file['correct']
        hum_phase[:, subj] = [int(ph[0]) for ph in feed_aliens_file['phase']]

        # Get rainbow data
        rainbow_file = subj_file[(subj_file['phase'] == '5RainbowSeason')].reset_index(drop=True)
        for trial in range(rainbow_file.shape[0]):
            alien, item = rainbow_file['sad_alien'][trial], rainbow_file['item_chosen'][trial]
            if not np.isnan(item):
                hum_rainbow_dat[subj, int(alien), int(item)] += 1

    hum_rainbow_dat = hum_rainbow_dat / np.sum(hum_rainbow_dat, axis=2, keepdims=True)  # Get fractions for every subj
    hum_rainbow_dat = np.mean(hum_rainbow_dat, axis=0)  # Average over subjects

    # Get competition data
    comp_file_names = [file_name for file_name in os.listdir(human_data_path) if "pick" in file_name]
    assert((n_hum == len(comp_file_names)), "The competition files and initial learning files are not for the same subjects!")

    hum_comp_dat = pd.DataFrame(np.zeros((n_hum, 21)))  # 21 = 3 (n_colums season) + 18 (n_columns season_alien)

    for subj, file_name in enumerate(comp_file_names):

        # Read in from disc
        subj_file = pd.read_csv(human_data_path + '/' + file_name,
                                usecols=['assess', 'id_chosen', 'id_unchosen', 'selected_better_obj'],
                                dtype={'assess': 'str', 'id_chosen': 'str', 'id_unchosen': 'str', 'selected_better_obj': 'float'}).reset_index(drop=True)
        subj_file = subj_file[np.invert(subj_file['id_unchosen'].isnull())]

        # Get season phase
        season_file = subj_file.loc[(subj_file['assess'] == 'season')]
        season_file.loc[(season_file['id_chosen'].astype(int) + season_file['id_unchosen'].astype(int)) == 1, 'choice'] = "(0, 1)"  # TODO: throws warning SettingWithCopyWarning:  A value is trying to be set on a copy of a slice from a DataFrame. Try using .loc[row_indexer,col_indexer] = value instead
        season_file.loc[(season_file['id_chosen'].astype(int) + season_file['id_unchosen'].astype(int)) == 2, 'choice'] = "(0, 2)"
        season_file.loc[(season_file['id_chosen'].astype(int) + season_file['id_unchosen'].astype(int)) == 3, 'choice'] = "(1, 2)"
        sum_season_file = season_file[['selected_better_obj', 'choice']].groupby('choice').aggregate('mean')

        # Get season-alien phase
        season_alien_file = subj_file.loc[(subj_file['assess'] == 'alien-same-season')]
        season_alien_file.loc[:, 'alien_a'] = season_alien_file['id_unchosen'].str[0].astype(int)
        season_alien_file.loc[:, 'alien_b'] = season_alien_file['id_chosen'].str[0].astype(int)
        season_alien_file.loc[:, 'choice'] = season_alien_file['id_unchosen'].str[1] + "(" +\
                                             season_alien_file[['alien_a', 'alien_b']].min(axis=1).astype(str) + ", " +\
                                             season_alien_file[['alien_a', 'alien_b']].max(axis=1).astype(str) + ")"
        sum_season_alien_file = season_alien_file[['selected_better_obj', 'choice']].groupby('choice').aggregate('mean')

        # Add all subjects together
        comp = sum_season_file.append(sum_season_alien_file)
        hum_comp_dat.loc[subj] = comp.values.flatten()
        hum_comp_dat.columns = comp.index.values
        hum_comp_dat.loc[:, '2(1, 2)'] = np.nan  # aliens 1 and 2 have the same value in TS 2 -> select better is not defined!

    return n_hum, hum_aliens, hum_seasons, hum_corrects, hum_actions, hum_rainbow_dat, hum_comp_dat


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
