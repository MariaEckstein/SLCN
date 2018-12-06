import datetime
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import pandas as pd
from scipy import stats
from sklearn import manifold, datasets

from AlienTask import Task
from shared_aliens import alien_initial_Q, update_Qs_sim, get_alien_paths,\
    get_summary_initial_learn, get_summary_cloudy, simulate_competition_phase, simulate_rainbow_phase, get_summary_rainbow,\
    read_in_human_data


# Define things
run_on_cluster = False
do_calculate_summaries = False
do_read_in_and_visualize_summaries = True
do_analyze_humans = False
model_name = 'hier'
models = ['flat', 'hier']
n_iter = 1000
n_sim_per_subj, n_subj = 10, 31  # n_sim_per_sub = 20, n_subj = 31 (version3.1)
n_sim = n_sim_per_subj * n_subj
n_actions, n_aliens, n_seasons, n_TS = 3, 4, 3, 3
human_data_path = get_alien_paths()["human data prepr"]
if model_name == 'hier':
    param_names = ['alpha', 'beta', 'forget', 'alpha_high', 'beta_high', 'forget_high']
    param_ranges = pd.DataFrame.from_dict(
        {'alpha': [0, 1], 'beta': [1, 20], 'forget': [0, 1],
         'alpha_high': [0, 1], 'beta_high': [1, 20], 'forget_high': [0, 1]
         })
elif model_name == 'flat':
    param_names = ['alpha', 'beta', 'forget']
    param_ranges = pd.DataFrame.from_dict({'alpha': [0, 1], 'beta': [1, 20], 'forget': [0, 1]})
else:
    raise(NameError, 'model_name must be "flat" or "hier"!')

summary_dat_cols = param_names + [
    'IL_saving_av', 'IL_saving_first_trial', 'IL_saving_last_trial',  # savings
    'IL_acc_current_TS', 'IL_acc_prev_TS', 'IL_acc_other_TS',  # intrusion errors
    'IL_perf_TS0', 'IL_perf_TS1', 'IL_perf_TS2', 'IL_perf_TS_corr'] + [  # TS values
    'CL_acc_trial0', 'CL_acc_trial1', 'CL_acc_trial2', 'CL_acc_trial3', 'CL_corr'] + [  # TS reactivation
    'CO_acc_season', 'CO_acc_season_alien'] + [  # competition alien values & TS values
    'RB_alien0_action0', 'RB_alien0_action1', 'RB_alien0_action2',
    'RB_alien1_action0', 'RB_alien1_action1', 'RB_alien1_action2',
    'RB_alien2_action0', 'RB_alien2_action1', 'RB_alien2_action2',
    'RB_alien3_action0', 'RB_alien3_action1', 'RB_alien3_action2',
]
plot_dir = get_alien_paths(run_on_cluster)['fitting results'] + '/SummariesInsteadOfFitting/'
now = datetime.datetime.now()
save_id = '{0}_{1}_{2}_{3}'.format(model_name, param_names, [str(i) for i in np.asarray(param_ranges)], '_'.join([str(i) for i in [now.year, now.month, now.day, now.hour, now.minute]]))


# Create task
task = Task(n_subj)
n_trials, _, _ = task.get_trial_sequence(get_alien_paths(run_on_cluster)["human data prepr"],
                                         n_subj, n_sim_per_subj, range(n_subj),
                                         phases=("1InitialLearning", "2CloudySeason"))

n_trials_ = {'1InitialLearn': np.sum(task.phase == '1InitialLearning'),
             '2CloudySeason': np.sum(task.phase == '2CloudySeason'),
             '4RainbowSeason': 4 * n_aliens}
trials = {'1InitialLearn': range(n_trials_['1InitialLearn']),
          '2CloudySeason': range(n_trials_['1InitialLearn'],
                                 n_trials_['1InitialLearn'] + n_trials_['2CloudySeason']),
          '4RainbowSeason': range(n_trials_['2CloudySeason'],
                                  n_trials_['2CloudySeason'] + n_trials_['4RainbowSeason'])}

# Get human data
if do_analyze_humans:
    n_hum, hum_aliens, hum_seasons, hum_corrects, hum_actions, hum_rainbow_dat, hum_comp_dat = read_in_human_data(human_data_path, 828, n_aliens, n_actions)

    # Get agent-like summaries
    hum_summary_initial_learn = get_summary_initial_learn(hum_seasons[trials['1InitialLearn']], hum_corrects[trials['1InitialLearn']], hum_aliens[trials['1InitialLearn']], hum_actions[trials['1InitialLearn']],
                                                          n_seasons, n_hum, trials, task)
    hum_summary_cloudy = get_summary_cloudy(hum_seasons, hum_corrects, n_hum, trials['2CloudySeason'])
    hum_summary_competition = np.mean(hum_comp_dat, axis=0)
    hum_summary_rainbow = get_summary_rainbow(n_aliens, n_seasons, hum_rainbow_dat, task)

    # Test for significance in human competition phase
    season_cols = [col for col in hum_comp_dat.columns.values if col[0] == "("]
    alien_cols = [col for col in hum_comp_dat.columns.values if col[0] != "("]
    season_perf = np.mean(hum_comp_dat[season_cols], axis=1)
    alien_perf = np.mean(hum_comp_dat[alien_cols], axis=1)
    comp_t, comp_p = stats.ttest_rel(season_perf, alien_perf)

    # Plot human behavior
    fig = plt.figure()
    ax = fig.add_subplot(331)
    plt.title('Initial learning phase')
    ax.bar(np.arange(3), hum_summary_initial_learn[:3])
    ax.set_xticks(range(3))
    ax.set_xticklabels(['savings (av.)', 'savings (1st)', 'savings (last)'])
    ax.set_ylabel('Savings')
    plt.xticks(rotation=20)

    ax = fig.add_subplot(332)
    plt.title('Initial learning phase')
    ax.bar(np.arange(3), hum_summary_initial_learn[3:6])
    ax.set_xticks(range(3))
    ax.axhline(y=1/3, color='grey', linestyle='--')
    ax.set_xticklabels(['acc (current)', 'acc (prev)', 'acc (other)'])
    ax.set_ylabel('Intrusion errors')
    plt.xticks(rotation=20)

    ax = fig.add_subplot(333)
    plt.title('Initial learning phase')
    ax.bar(np.arange(3), hum_summary_initial_learn[6:9])
    ax.set_xticks(range(3))
    ax.set_xticklabels(['TS0', 'TS1', 'TS2'])
    ax.set_ylabel('TS performance (r={})'.format(hum_summary_initial_learn[9].round(2)))
    plt.xticks(rotation=20)

    ax = fig.add_subplot(334)
    plt.title('Cloudy phase')
    ax.bar(np.arange(4), hum_summary_cloudy[:4])
    ax.axhline(y=1/3, color='grey', linestyle='--')
    ax.set_ylabel('ACC (r={})'.format(hum_summary_cloudy[4].round(2)))
    ax.set_xticks(range(4))
    ax.set_xticklabels(['0 seen', '1 seen', '2 seen', '3 seen'])
    plt.xticks(rotation=20)

    ax = fig.add_subplot(335)
    plt.title('Rainbow phase')
    ax.matshow(hum_rainbow_dat.T)
    ax.set_xlabel('Aliens')
    ax.set_ylabel('Actions')

    ax = fig.add_subplot(336)
    plt.title('Rainbow phase')
    ax.bar(np.arange(4), hum_summary_rainbow[:4])
    ax.set_xticks(range(4))
    ax.set_xticklabels(['TS0', 'TS1', 'TS2', 'None'])
    ax.set_ylabel('% chosen (slope={})'.format(hum_summary_rainbow[4].round(2)))

    ax = fig.add_subplot(337)
    plt.title('Competition phase')
    ax.bar(hum_summary_competition.index.values, hum_summary_competition.values)
    ax.axhline(y=1/2, color='grey', linestyle='--')
    ax.set_ylabel('% better chosen (p={})'.format(comp_p.round(3)))
    plt.xticks(rotation=20)

    plt.tight_layout()


# Function to calculate summaries
def get_summary(parameters, param_ranges, n_sim, n_subj):

    # Get parameters
    parameters = param_ranges.loc[0] + (param_ranges.loc[1] - param_ranges.loc[0]) * parameters

    beta_shape = (n_sim, 1)  # Q_low_sub.shape -> [n_subj, n_actions]
    beta_high_shape = (n_sim, 1)  # Q_high_sub.shape -> [n_subj, n_TS]
    forget_high_shape = (n_sim, 1, 1)  # -> [n_subj, n_seasons, n_TS]
    forget_shape = (n_sim, 1, 1, 1)  # Q_low[0].shape -> [n_subj, n_TS, n_aliens, n_actions]

    alpha = parameters['alpha'] * np.ones(n_sim)
    beta = parameters['beta'] * np.ones(beta_shape)
    forget = parameters['forget'] * np.ones(forget_shape)
    try:
        alpha_high = parameters['alpha_high'] * np.ones(n_sim)
        beta_high = parameters['beta_high'] * np.ones(beta_high_shape)
        forget_high = parameters['forget_high'] * np.ones(forget_high_shape)
    except KeyError:  # if 'alpha_high' not in parameters
        alpha_high = 0.1 * np.ones(n_sim)
        beta_high = np.ones(beta_high_shape)
        forget_high = np.zeros(forget_high_shape)

    # Initial learning phase
    seasons = np.zeros([n_trials, n_sim], dtype=int)
    corrects = np.zeros([n_trials, n_sim])
    rewards = np.zeros([n_trials, n_sim])
    aliens = np.zeros([n_trials, n_sim], dtype=int)
    actions = np.zeros([n_trials, n_sim], dtype=int)

    Q_low = alien_initial_Q * np.ones([n_sim, n_TS, n_aliens, n_actions])
    Q_high = alien_initial_Q * np.ones([n_sim, n_seasons, n_TS])

    for trial in trials['1InitialLearn']:

        # Observe stimuli
        season, alien = task.present_stimulus(trial)

        # Select action & update Q-values
        [Q_low, Q_high, TS, action, correct, reward, p_low] =\
            update_Qs_sim(season, alien,
                          Q_low, Q_high,
                          beta, beta_high, alpha, alpha_high, forget, forget_high,
                          n_sim, n_actions, n_TS, task, verbose=False)

        # Store trial data
        seasons[trial] = season
        corrects[trial] = correct
        rewards[trial] = reward
        aliens[trial] = alien
        actions[trial] = action

    summary_initial_learn = get_summary_initial_learn(seasons[trials['1InitialLearn']], corrects[trials['1InitialLearn']], aliens[trials['1InitialLearn']], actions[trials['1InitialLearn']],
                                                      n_seasons, n_sim, trials, task)

    # Save final Q-values for subsequent phases
    final_Q_low = Q_low.copy()
    final_Q_high = Q_high.copy()

    # Cloudy season
    for trial in trials['2CloudySeason']:

        # Observe stimuli
        old_season = season.copy()
        season, alien = task.present_stimulus(trial)

        # Take care of season switches
        if trial == list(trials['2CloudySeason'])[0]:
            season_switches = np.ones(n_sim, dtype=bool)
        else:
            season_switches = season != old_season

        if model_name == 'hier':
            Q_high[season_switches] = alien_initial_Q  # re-start search for the right TS when season changes
        elif model_name == 'flat':
            Q_low[season_switches] = alien_initial_Q  # re-learn a new policy from scratch when season changes
        else:
            raise(NameError, 'Model_name must be either "flat" or "hier".')

        # Update Q-values
        [Q_low, Q_high, TS, action, correct, reward, p_low] =\
            update_Qs_sim(0 * season, alien,
                          Q_low, Q_high,
                          beta, beta_high, alpha, alpha_high, forget, forget_high,
                          n_sim, n_actions, n_TS, task)

        # Store trial data
        seasons[trial] = season
        corrects[trial] = correct

    summary_cloudy = get_summary_cloudy(seasons, corrects, n_sim, trials['2CloudySeason'])

    # Run competition phase
    comp_data = simulate_competition_phase(model_name, final_Q_high, final_Q_low, task,
                                           n_seasons, n_aliens, n_sim, beta_high)
    summary_competition = comp_data.groupby('phase').aggregate('mean')['perc_selected_better'].values

    # Run rainbow season
    rainbow_dat = simulate_rainbow_phase(n_seasons, model_name, n_sim,
                                         beta, beta_high, final_Q_low, final_Q_high)
    TS_choices = get_summary_rainbow(n_aliens, n_seasons, rainbow_dat, task)
    summary_rainbow = np.mean(TS_choices, axis=1)

    return list(parameters.values) + summary_initial_learn + list(summary_cloudy) + list(summary_competition) + list(rainbow_dat.flatten())


# Get summaries for different parameters
if do_calculate_summaries:
    summaries = pd.DataFrame(np.full((n_iter, len(summary_dat_cols)), np.nan), columns=summary_dat_cols)
    for iter in range(n_iter):
        print("Iteration {}".format(iter))

        params = np.random.rand(len(param_names))
        # params = [0.01, 6.5/20, 0.005, 0.3, 4.5/20, 0.005]  # TODO: debug - remove!
        # params = [0.1, 5, 0]
        summaries.loc[iter] = get_summary(params, param_ranges, n_sim, n_subj)

        # Save summaries to disk (every trial)
        save_path = plot_dir + save_id + '_summaries.csv'
        print("Saving summaries to {}".format(save_path))
        summaries.to_csv(save_path)

    print(summaries.agg('mean'))
    print(summaries)

# Vizualize
if do_read_in_and_visualize_summaries:

    # Combine all csvs
    filenames = glob.glob(os.path.join(plot_dir, '*.csv'))
    print('Reading in {} files'.format(len(filenames)))
    all_summaries = pd.DataFrame(columns=param_names)
    for filename in filenames:
        summaries = pd.read_csv(filename, index_col=0)
        summaries = summaries.dropna()
        all_summaries = all_summaries.append(summaries)
    all_summaries = all_summaries.reset_index(drop=True)

    # Add model column (flat or hier)
    all_summaries['model'] = 'hier'
    all_summaries.loc[all_summaries.isnull()['alpha_high'], 'model'] = 'flat'

    print("Number of samples: {0} (flat: {1}; hier: {2})".
          format(all_summaries.shape[0],
                 np.sum(all_summaries['model'] == 'flat'),
                 np.sum(all_summaries['model'] == 'hier')))

    # Add other measures
    all_summaries['IL_saving_last_minus_first'] = all_summaries['IL_saving_last_trial'] - all_summaries['IL_saving_first_trial']
    all_summaries['CO_season_minus_alien'] = all_summaries['CO_acc_season'] - all_summaries['CO_acc_season_alien']

    # Plot correlations and histograms
    for model_name in models:
        model_summaries = all_summaries.loc[all_summaries['model'] == model_name]
        model_summaries = model_summaries.reset_index(drop=True)
        pd.scatter_matrix(model_summaries.loc[:, ['alpha', 'beta', 'forget']])

    # Plot savings (Initial Learning)
    plt.figure()
    for i, (effect, name) in enumerate(zip(['IL_saving_first_trial', 'IL_saving_last_trial', 'IL_saving_av', 'IL_saving_last_minus_first'],
                                           ['on first trial', 'on last trial', 'on average', 'last minus first'])):
        plt.subplot(2, 2, i+1)
        bins = np.arange(min(all_summaries[effect]), max(all_summaries[effect]), 0.01)
        plt.axvline(x=0, color='grey', linestyle='--')
        plt.ylim(0, 25)
        plt.xlim(-0.1, 0.5)
        for model in models:
            dat = all_summaries.loc[all_summaries['model'] == model]
            plt.hist(dat[effect], bins=bins, alpha=0.5, label=model, density=True)
        plt.xlabel("Savings {}".format(name))
        plt.ylabel("Probability density")
    plt.legend()
    plt.tight_layout()

    # Plot intrusion errors (Initial Learning)
    plt.figure()
    for i, (effect, name) in enumerate(zip(['IL_acc_prev_TS', 'IL_acc_other_TS', 'IL_acc_current_TS'],
                                           ['Intrusion errors previous TS', 'Intrusion errors other TS', '% correct'])):
        plt.subplot(2, 2, i+1)
        bins = np.arange(min(all_summaries[effect]), max(all_summaries[effect]), 0.01)
        plt.axvline(x=1/3, color='grey', linestyle='--')
        plt.xlim(0, 1)
        plt.ylim(0, 10)
        for model in models:
            dat = all_summaries.loc[all_summaries['model'] == model]
            plt.hist(dat[effect], bins=bins, alpha=0.5, label=model, density=True)
        plt.xlabel(name)
        plt.ylabel("Probability density")
    plt.legend()
    plt.tight_layout()

    # Plot TS values (Initial Learning)
    plt.figure()
    for i, (effect, name) in enumerate(zip(['IL_perf_TS0', 'IL_perf_TS1', 'IL_perf_TS2'],
                                           ['TS0', 'TS1', 'TS2'])):
        plt.subplot(2, 2, i+1)
        bins = np.arange(min(all_summaries[effect]), max(all_summaries[effect]), 0.01)
        plt.xlim(0, 1)
        # plt.ylim(0, 10)
        for model in models:
            dat = all_summaries.loc[all_summaries['model'] == model]
            plt.hist(dat[effect], bins=bins, alpha=0.5, label=model, density=True)
        plt.xlabel("ACC {}".format(name))
        plt.ylabel("Probability density")
    plt.legend()

    plt.subplot(2, 2, 4)
    bins = np.arange(-1, -0.8, 0.005)
    effect = 'IL_perf_TS_corr'
    for model in models:
        dat = all_summaries.loc[all_summaries['model'] == model]
        plt.hist(dat[effect], bins=bins, alpha=0.5, label=model, density=True)
    plt.xlabel("Correlation (r)")
    plt.ylabel("Probability density")
    plt.legend()
    plt.tight_layout()

    # Plot TS reactivation (Cloudy Season)
    plt.figure()
    for i, effect in enumerate(['CL_acc_trial0', 'CL_acc_trial1', 'CL_acc_trial2', 'CL_acc_trial3']):
        plt.subplot(3, 2, i+1)
        bins = np.arange(min(all_summaries[effect]), max(all_summaries[effect]), 0.01)
        plt.axvline(x=1/3, color='grey', linestyle='--')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        for model in models:
            dat = all_summaries.loc[all_summaries['model'] == model]
            plt.hist(dat[effect], bins=bins, alpha=0.5, label=model, density=True)
        plt.xlabel("Accuracy after seeing {} other aliens".format(effect[-1]))
        plt.ylabel("Probability density")
    plt.legend()
    plt.tight_layout()

    plt.subplot(3, 2, 5)
    effect = 'CL_corr'
    bins = np.arange(min(all_summaries[effect]), max(all_summaries[effect]), 0.01)
    for model in models:
        dat = all_summaries.loc[all_summaries['model'] == model]
        plt.hist(dat[effect], bins=bins, alpha=0.5, label=model, density=True)
    plt.axvline(x=0, color='grey', linestyle='--')
    plt.xlabel("Increase in accuracy (r)")
    plt.ylabel("Probability density")
    plt.legend()

    # Competition phase
    plt.figure()
    for i, effect in enumerate(['CO_acc_season', 'CO_acc_season_alien']):
        plt.subplot(3, 1, i+1)
        bins = np.arange(min(all_summaries[effect]), max(all_summaries[effect]), 0.01)
        plt.axvline(x=1/2, color='grey', linestyle='--')
        plt.xlim(0, 1)
        for model in models:
            dat = all_summaries.loc[all_summaries['model'] == model]
            plt.hist(dat[effect], bins=bins, alpha=0.5, label=model, density=True)
        plt.xlabel("% selecting better {}".format(effect[7:]))
        plt.ylabel("Probability density")
    plt.legend()

    effect = 'CO_season_minus_alien'
    plt.subplot(3, 1, 3)
    bins = np.arange(min(all_summaries[effect]), max(all_summaries[effect]), 0.01)
    plt.axvline(x=0, color='grey', linestyle='--')
    for model in models:
        dat = all_summaries.loc[all_summaries['model'] == model]
        plt.hist(dat[effect], bins=bins, alpha=0.5, label=model, density=True)
    plt.xlabel("Season minus season-alien")
    plt.ylabel("Probability density")
    plt.legend()
    plt.tight_layout()

    # Rainbow phase TS choices
    rainbow_columns = [col for col in all_summaries.columns.values if 'RB_alien' in col]
    plt.figure()
    for model in models:
        rainbow_dat = all_summaries.loc[all_summaries['model'] == model, rainbow_columns]
        rainbow_dat = rainbow_dat.as_matrix().reshape((rainbow_dat.shape[0], n_aliens, n_actions))

        summary_rainbow = np.array([get_summary_rainbow(n_aliens, n_seasons, dat, task) for dat in rainbow_dat])

        bins = np.arange(0, 0.5, 0.01)
        for i, TS in enumerate(['TS0', 'TS1', 'TS2', 'None']):
            plt.subplot(2, 3, i+1)
            plt.hist(summary_rainbow[:, i], alpha=0.5, bins=bins, label=model, density=True)
            plt.axvline(x=1/4, color='grey', linestyle='--')
            plt.ylim(0, 15)
            plt.xlabel("Count of choosing " + TS)
            plt.ylabel("Probability density")

        plt.subplot(2, 3, 5)
        bins = np.arange(-0.2, 0.2, 0.01)
        plt.hist(summary_rainbow[:, 4], alpha=0.5, bins=bins, label=model, density=True)
        plt.axvline(x=0, color='grey', linestyle='--')
        plt.ylim(0, 15)
        plt.xlabel("Slope")
        plt.ylabel("Probability density")

    plt.legend()
    plt.tight_layout()

    # Intrusion error heatmap (prev_TS)
    fig = plt.figure()
    for i, effect in enumerate(['IL_acc_prev_TS', 'IL_acc_current_TS']):

        dat = all_summaries.loc[(all_summaries['model'] == 'hier') &
                                (all_summaries['IL_acc_prev_TS'] > 0.55)]

        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        ax.set_title(effect)
        ax.scatter(dat['alpha'], dat['beta'], dat['forget'], c=dat[effect], label=dat[effect], marker='.')
        ax.set_xlabel('alpha')
        ax.set_ylabel('beta')
        ax.set_zlabel('forget')

        ax = fig.add_subplot(2, 2, i+3, projection='3d')
        ax.set_title(effect)
        ax.scatter(dat['alpha_high'], dat['beta_high'], dat['forget_high'], c=dat[effect], label=dat[effect], marker='.')
        ax.set_xlabel('alpha_high')
        ax.set_ylabel('beta_high')
        ax.set_zlabel('forget_high')
    plt.tight_layout()

    # Calculate Isomap and visualize
    params = dat[['alpha', 'beta', 'forget', 'alpha_high', 'beta_high', 'forget_high']]
    effect = dat['IL_acc_prev_TS']
    fig = plt.figure(figsize=(15, 5))

    i = 1
    for n_components in range(2, 6, 2):
        for n_neighbors in range(2, 20, 3):
            # Get new manifold
            # Y = manifold.LocallyLinearEmbedding(n_neighbors, n_components, eigen_solver='auto',
            #                                     method='standard').fit_transform(params)
            # Y = manifold.LocallyLinearEmbedding(n_neighbors, n_components, eigen_solver='auto',
            #                                     method='modified').fit_transform(params)
            # mds = manifold.MDS(n_neighbors, max_iter=100, n_init=1)
            # Y = mds.fit_transform(params)
            Y = manifold.Isomap(n_neighbors, n_components).fit_transform(params)

            # Plot
            ax = fig.add_subplot(2, 6, i)
            plt.scatter(Y[:, 0], Y[:, 1], c=effect)
            plt.title('{0} comp., {1} neigh.'.format(n_components, n_neighbors))
            i += 1
    plt.tight_layout()

    # Intrusion error heatmap (other TS)
    fig = plt.figure()
    for i, effect in enumerate(['IL_acc_other_TS', 'IL_acc_current_TS']):

        dat = all_summaries.loc[(all_summaries['model'] == 'hier') &
                                (all_summaries['IL_acc_other_TS'] > 1/3)]

        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        ax.set_title(effect)
        ax.scatter(dat['alpha'], dat['beta'], dat['forget'], c=dat[effect], label=dat[effect], marker='.')
        ax.set_xlabel('alpha')
        ax.set_ylabel('beta')
        ax.set_zlabel('forget')

        ax = fig.add_subplot(2, 2, i+3, projection='3d')
        ax.set_title(effect)
        ax.scatter(dat['alpha_high'], dat['beta_high'], dat['forget_high'], c=dat[effect], label=dat[effect], marker='.')
        ax.set_xlabel('alpha_high')
        ax.set_ylabel('beta_high')
        ax.set_zlabel('forget_high')
    plt.tight_layout()

    # Savings heatmap
    fig = plt.figure()
    for i, effect in enumerate(['IL_saving_av', 'IL_saving_first_trial', 'IL_saving_last_trial', 'IL_saving_last_minus_first']):

        dat = all_summaries.loc[(all_summaries['model'] == 'hier') &
                                (all_summaries['IL_saving_last_trial'] > 0.01) & (all_summaries['IL_saving_first_trial'] > 0.08)]

        ax = fig.add_subplot(2, 4, i+1, projection='3d')
        ax.set_title(effect)
        ax.scatter(dat['alpha'], dat['beta'], dat['forget'], c=dat[effect], label=dat[effect], marker='.')
        ax.set_xlabel('alpha')
        ax.set_ylabel('beta')
        ax.set_zlabel('forget')

        ax = fig.add_subplot(2, 4, i+5, projection='3d')
        ax.set_title(effect)
        ax.scatter(dat['alpha_high'], dat['beta_high'], dat['forget_high'], c=dat[effect], label=dat[effect], marker='.')
        ax.set_xlabel('alpha_high')
        ax.set_ylabel('beta_high')
        ax.set_zlabel('forget_high')
    plt.tight_layout()

    # Cloudy heatmap
    fig = plt.figure()
    for i, effect in enumerate(['CL_acc_trial2', 'CL_acc_trial3']):

        dat = all_summaries.loc[(all_summaries['model'] == 'hier') &
                                (all_summaries['CL_acc_trial2'] > 0.4) & (all_summaries['CL_acc_trial3'] > 0.4)]

        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        ax.set_title(effect)
        ax.scatter(dat['alpha'], dat['beta'], dat['forget'], c=dat[effect], label=dat[effect], marker='.')
        ax.set_xlabel('alpha')
        ax.set_ylabel('beta')
        ax.set_zlabel('forget')

        ax = fig.add_subplot(2, 2, i+3, projection='3d')
        ax.set_title(effect)
        ax.scatter(dat['alpha_high'], dat['beta_high'], dat['forget_high'], c=dat[effect], label=dat[effect], marker='.')
        ax.set_xlabel('alpha_high')
        ax.set_ylabel('beta_high')
        ax.set_zlabel('forget_high')
    plt.tight_layout()

    # Competition heatmap
    fig = plt.figure()
    for i, effect in enumerate(['CO_acc_season', 'CO_acc_season_alien']):

        dat = all_summaries.loc[(all_summaries['model'] == 'hier') &
                                (all_summaries['CO_acc_season'] > 0.65)]  #  &(all_summaries['CO_acc_season_alien'] > 0.60)

        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        ax.set_title(effect)
        ax.scatter(dat['alpha'], dat['beta'], dat['forget'], c=dat[effect], label=dat[effect], marker='.')
        ax.set_xlabel('alpha')
        ax.set_ylabel('beta')
        ax.set_zlabel('forget')

        ax = fig.add_subplot(2, 2, i+3, projection='3d')
        ax.set_title(effect)
        ax.scatter(dat['alpha_high'], dat['beta_high'], dat['forget_high'], c=dat[effect], label=dat[effect], marker='.')
        ax.set_xlabel('alpha_high')
        ax.set_ylabel('beta_high')
        ax.set_zlabel('forget_high')
    plt.tight_layout()

    # Rainbow heatmap
    fig = plt.figure()
    for i, effect in enumerate(['RB_choices_TS0', 'RB_choices_TS1', 'RB_choices_TS2']):

        dat = all_summaries.loc[(all_summaries['model'] == 'hier') &
                                (all_summaries['RB_choices_TS0'] > 1300) & (all_summaries['RB_choices_TS1'] > 1000)]

        ax = fig.add_subplot(2, 3, i+1, projection='3d')
        ax.set_title(effect)
        ax.scatter(dat['alpha'], dat['beta'], dat['forget'], c=dat[effect], label=dat[effect], marker='.')
        ax.set_xlabel('alpha')
        ax.set_ylabel('beta')
        ax.set_zlabel('forget')

        ax = fig.add_subplot(2, 3, i+4, projection='3d')
        ax.set_title(effect)
        ax.scatter(dat['alpha_high'], dat['beta_high'], dat['forget_high'], c=dat[effect], label=dat[effect], marker='.')
        ax.set_xlabel('alpha_high')
        ax.set_ylabel('beta_high')
        ax.set_zlabel('forget_high')
    plt.tight_layout()

    # Plot overall heatmaps -> systematicity between parameters & effects?
    for effect in summary_dat_cols:
        fig = plt.figure()
        plt.title(effect)
        for i, model in enumerate(models):
            dat = all_summaries.loc[all_summaries['model'] == model][:1000].copy()

            ax = fig.add_subplot(2, 2, i+1, projection='3d')
            ax.scatter(dat['alpha'], dat['beta'], dat['forget'], c=dat[effect], label=dat[effect], marker='.')
            ax.set_title(model)
            ax.set_xlabel('alpha')
            ax.set_ylabel('beta')
            ax.set_zlabel('forget')

            ax = fig.add_subplot(2, 2, i+3, projection='3d')
            ax.scatter(dat['alpha_high'], dat['beta_high'], dat['forget_high'], c=dat[effect], label=dat[effect], marker='.')
            ax.set_xlabel('alpha_high')
            ax.set_ylabel('beta_high')
            ax.set_zlabel('forget_high')
        plt.tight_layout()

    # Plot average effects for hierarchical and flat
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i, model in enumerate(models):
        dat = all_summaries.loc[all_summaries['model'] == model].copy()
        RB_effects = [effect for effect in all_summaries.columns.values if 'RB' in effect]

        dat[RB_effects] /= 1000
        dat['beta'] /= 10
        x = np.arange(len(summary_dat_cols)) + 0.2 * (2*i-1)
        y = np.mean(dat[summary_dat_cols], axis=0)
        yerr = np.std(dat[summary_dat_cols], axis=0) / np.sqrt(len(summary_dat_cols))

        ax.bar(x, y.values, 0.4, yerr=yerr, label=model)
    ax.set_xticks(np.arange(len(summary_dat_cols)))
    ax.set_xticklabels(summary_dat_cols)
    plt.xticks(rotation=20)
    plt.ylabel('Effect (RB /= 1000)')
    plt.legend()



stop = 3
