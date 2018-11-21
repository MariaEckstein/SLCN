import glob
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import pandas as pd

from AlienTask import Task
from shared_aliens import alien_initial_Q, update_Qs_sim, get_alien_paths,\
    get_summary_initial_learn, get_summary_cloudy, simulate_competition_phase, simulate_rainbow_phase, get_summary_rainbow


# Define things
run_on_cluster = False
do_calculate_summaries = True
do_read_in_and_visualize_summaries = False
model_name = 'flat'
models = ['flat', 'hier']
n_iter = 1000
n_sim_per_subj, n_subj = 20, 31  # n_sim_per_sub = 20, n_subj = 31 (version3.1)
n_sim = n_sim_per_subj * n_subj
n_actions, n_aliens, n_seasons, n_TS = 3, 4, 3, 3
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
    'IL_saving_av', 'IL_saving_first_trial', 'IL_saving_last_trial', 'IL_acc_current_TS', 'IL_acc_prev_TS', 'IL_acc_other_TS'] + [
    'CL_acc_trial0', 'CL_acc_trial1', 'CL_acc_trial2', 'CL_acc_trial3'] + [
    'CO_acc_season', 'CO_acc_season_alien'] + [
    'RB_choices_TS0', 'RB_choices_TS1', 'RB_choices_TS2', 'RB_choices_none'
]
plot_dir = get_alien_paths(run_on_cluster)['fitting results'] + '/SummariesInsteadOfFitting/'
now = datetime.datetime.now()
save_id = '{0}_{1}_{2}_{3}'.format(model_name, param_names, [str(i) for i in np.asarray(param_ranges)], '_'.join([str(i) for i in [now.year, now.month, now.day, now.hour, now.minute]]))


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

    # Initial learning phase
    seasons = np.zeros([n_trials, n_sim], dtype=int)
    corrects = np.zeros([n_trials, n_sim])
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
        aliens[trial] = alien
        actions[trial] = action

    summary_initial_learn = get_summary_initial_learn(seasons, corrects, aliens, actions,
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

    summary_cloudy = get_summary_cloudy(seasons, corrects, n_seasons, n_sim, trials)

    # Run rainbow season
    rainbow_dat = simulate_rainbow_phase(n_seasons, n_aliens, n_actions, n_TS,
                                         model_name, n_sim,
                                         beta, beta_high, final_Q_low, final_Q_high)
    TS_choices = get_summary_rainbow(n_aliens, n_seasons, rainbow_dat, task)
    summary_rainbow = np.mean(TS_choices, axis=1)

    # Run competition phase
    comp_data = simulate_competition_phase(model_name, final_Q_high, final_Q_low, task,
                                           n_seasons, n_aliens, n_sim, beta_high)
    summary_competition = comp_data.groupby('phase').aggregate('mean')['perc_selected_better'].values

    return list(parameters.values) + summary_initial_learn + list(summary_cloudy) + list(summary_competition) + list(summary_rainbow)


# Get summaries for different parameters
if do_calculate_summaries:
    summaries = pd.DataFrame(np.full((n_iter, len(summary_dat_cols)), np.nan), columns=summary_dat_cols)
    for iter in range(n_iter):
        print("Iteration {}".format(iter))

        params = np.random.rand(len(param_names))
        # params = [0.01, 6.5/20, 0.005, 0.3, 4.5/20, 0.2]  # TODO: debug - remove!
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

    # Add model column (flat or hier)
    all_summaries['model'] = 'hier'
    all_summaries.loc[all_summaries.isnull()['alpha_high'], 'model'] = 'flat'

    print("Number of samples: {0} (flat: {1}; hier: {2})".
          format(all_summaries.shape[0],
                 np.sum(all_summaries['model'] == 'flat'),
                 np.sum(all_summaries['model'] == 'hier')))

    # Plot average effects for hierarchical and flat
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i, model in enumerate(models):
        dat = all_summaries.loc[all_summaries['model'] == model]
        RB_effects = [effect for effect in all_summaries.columns.values if 'RB' in effect]

        dat[RB_effects] /= 1000
        x = np.arange(len(summary_dat_cols)) + (2*i-1)/3
        y = np.mean(dat[summary_dat_cols], axis=0)
        yerr = np.std(dat[summary_dat_cols], axis=0) / np.sqrt(len(summary_dat_cols))

        ax.bar(x, y.values, yerr=yerr, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(summary_dat_cols)
    plt.xticks(rotation=20)
    plt.ylabel('Effect (RB /= 1000)')
    plt.legend()

    # Plot heatmaps
    for effect in summary_dat_cols:
        fig = plt.figure()
        plt.title(effect)
        for i, model in enumerate(models):
            dat = all_summaries.loc[all_summaries['model'] == model][:1000]

            ax = fig.add_subplot(2, 2, i + 1, projection='3d')
            ax.scatter(dat['alpha'], dat['beta'], dat['forget'], c=dat[effect], label=dat[effect], marker='.')
            ax.set_title(model)
            ax.set_xlabel('alpha')
            ax.set_ylabel('beta')
            ax.set_zlabel('forget')

            ax = fig.add_subplot(2, 2, i + 3, projection='3d')
            ax.scatter(dat['alpha_high'], dat['beta_high'], dat['forget_high'], c=dat[effect], label=dat[effect], marker='.')
            ax.set_xlabel('alpha_high')
            ax.set_ylabel('beta_high')
            ax.set_zlabel('forget_high')
        plt.tight_layout()
    # plt.show()

stop = 3
