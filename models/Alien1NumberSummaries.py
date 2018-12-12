import datetime
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import pandas as pd
from pandas.plotting import scatter_matrix
from scipy import stats

from AlienTask import Task
from shared_aliens import alien_initial_Q, update_Qs_sim, get_alien_paths,\
    get_summary_initial_learn, get_summary_cloudy, simulate_competition_phase, simulate_rainbow_phase, get_summary_rainbow,\
    read_in_human_data


# Define things
run_on_cluster = False
do_analyze_humans = True
do_calculate_summaries = False
do_read_in_and_visualize_summaries = True
do_isomap = False
do_save_selected_agents = False
if do_analyze_humans:
    import seaborn as sns
if do_isomap:
    from sklearn import manifold, decomposition, preprocessing
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

IL_cols = ['IL_saving_av', 'IL_saving_first_trial', 'IL_saving_last_trial',  # savings
           'IL_acc_current_TS', 'IL_acc_prev_TS', 'IL_acc_other_TS',  # intrusion errors
           'IL_acc_current_TS_se', 'IL_acc_prev_TS_se', 'IL_acc_other_TS_se',
           'IL_perf_TS0', 'IL_perf_TS1', 'IL_perf_TS2',  # TS values
           'IL_perf_TS0_se', 'IL_perf_TS1_se', 'IL_perf_TS2_se', 'IL_perf_TS_corr'
           ]
CL_cols = ['CL_acc_trial0', 'CL_acc_trial1', 'CL_acc_trial2', 'CL_acc_trial3',
           'CL_acc_trial0_se', 'CL_acc_trial1_se', 'CL_acc_trial2_se', 'CL_acc_trial3_se',
           'CL_slope', 'CL_slope_TS0', 'CL_slope_TS1', 'CL_slope_TS2']  # TS reactivation
CO_cols = ['CO_acc_season', 'CO_acc_season_alien', 'CO_acc_season_se', 'CO_acc_season_alien_se']  # competition alien values & TS values
RB_cols = ['RB_alien0_action0', 'RB_alien0_action1', 'RB_alien0_action2',
           'RB_alien1_action0', 'RB_alien1_action1', 'RB_alien1_action2',
           'RB_alien2_action0', 'RB_alien2_action1', 'RB_alien2_action2',
           'RB_alien3_action0', 'RB_alien3_action1', 'RB_alien3_action2']
RB_sum_cols = ['TS0', 'TS1', 'TS2', 'None', 'TS0_se', 'TS1_se', 'TS2_se', 'None_se', 'TS2minusTS0']
summary_dat_cols = param_names + IL_cols + CL_cols + CO_cols[:2] + RB_cols

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
    selected_agents = pd.read_csv(plot_dir + 'rerun_selected_agents.csv')
    ag_summary = selected_agents.loc[0]  # np.mean(selected_agents, axis=0)

    # Get agent-like summaries
    hum_summary_initial_learn = get_summary_initial_learn(hum_seasons[trials['1InitialLearn']], hum_corrects[trials['1InitialLearn']], hum_aliens[trials['1InitialLearn']], hum_actions[trials['1InitialLearn']],
                                                          n_seasons, n_hum, trials, task)
    hum_summary_initial_learn = pd.DataFrame(np.expand_dims(np.array(hum_summary_initial_learn), axis=0),
                                             columns=IL_cols)

    hum_summary_cloudy = get_summary_cloudy(hum_seasons, hum_corrects, n_hum, trials['2CloudySeason'])
    hum_summary_cloudy = pd.DataFrame(np.expand_dims(np.array(hum_summary_cloudy), axis=0),
                                      columns=CL_cols)

    season_cols = [col for col in hum_comp_dat.columns.values if col[0] == "("]
    alien_cols = [col for col in hum_comp_dat.columns.values if col[0] != "("]
    season_perf = np.mean(hum_comp_dat[season_cols], axis=1)
    alien_perf = np.mean(hum_comp_dat[alien_cols], axis=1)
    season_mean = np.mean(season_perf)
    season_se = np.std(season_perf) / np.sqrt(n_hum)
    alien_mean = np.mean(alien_perf)
    alien_se = np.std(alien_perf) / np.sqrt(n_hum)
    comp_t, comp_p = stats.ttest_rel(season_perf, alien_perf)
    hum_summary_competition = pd.DataFrame(np.array([[season_mean, alien_mean, season_se, alien_se, np.mean(season_perf-alien_perf)]]),
                                           columns=CO_cols + ['CO_season_minus_alien'])

    # hum_rainbow_dat = pd.DataFrame(np.expand_dims(hum_rainbow_dat.flatten(), axis=0), columns=RB_cols)
    hum_summary_rainbow = get_summary_rainbow(n_aliens, n_seasons, hum_rainbow_dat, task)
    hum_summary_rainbow = pd.DataFrame(np.expand_dims(hum_summary_rainbow, axis=0),
                                       columns=RB_sum_cols)
    hum_summary_rainbow['TS2minusTS1'] = hum_summary_rainbow['TS2'] - hum_summary_rainbow['TS1']

    ag_rainbow_dat = ag_summary[RB_cols].values.reshape((n_aliens, n_actions))
    ag_summary_rainbow = get_summary_rainbow(n_aliens, n_seasons, ag_rainbow_dat, task)
    ag_summary_rainbow = pd.DataFrame(np.expand_dims(ag_summary_rainbow, axis=0),
                                       columns=RB_sum_cols)
    ag_summary_rainbow['TS2minusTS1'] = ag_summary_rainbow['TS2'] - ag_summary_rainbow['TS1']

    # Add other measures
    hum_summary_initial_learn['IL_saving_last_minus_first'] = hum_summary_initial_learn['IL_saving_last_trial'] - hum_summary_initial_learn['IL_saving_first_trial']
    hum_summary_initial_learn['IL_perf_TS2minus1'] = hum_summary_initial_learn['IL_perf_TS2'] - hum_summary_initial_learn['IL_perf_TS1']

    # Plot human behavior & selected_agents
    def make_plot(hum_mean, ag_mean, hum_se=False, ag_se=False, ylabel="", hline=False, ylim=False, xlabel="", xticklabels=False):
        if not np.any(hum_se):
            hum_se = np.zeros(len(hum_mean.T))
        if not np.any(ag_se):
            ag_se = np.zeros(len(ag_mean.T))
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
        [ax.set_title(title) for ax, title in zip(axes, ['Humans', 'Simulations'])]
        hum_mean.T.plot.bar(rot=0, legend=False, ax=axes[0], yerr=hum_se, color='darkblue')
        ag_mean.T.plot.bar(rot=0, legend=False, ax=axes[1], yerr=ag_se, color='darkblue')
        [ax.set_ylabel(ylabel) for ax in axes]
        [ax.set_xlabel(xlabel) for ax in axes]
        if xticklabels:
            [ax.set_xticklabels(xticklabels) for ax in axes]
        if ylim:
            [ax.set_ylim(ylim) for ax in axes]
        if hline:
            [ax.axhline(y=hline, color='grey', linestyle='--') for ax in axes]
        plt.tight_layout()

    # Reactivation of TS
    make_plot(hum_summary_cloudy[CL_cols[:4]], ag_summary[CL_cols[:4]],
              hum_summary_cloudy[CL_cols[4:8]].values, ag_summary[CL_cols[4:8]].values,
              ylabel="Accuracy (trial 1)", hline=1/3, ylim=(0, 0.45),
              xlabel="# other aliens seen", xticklabels=[str(i) for i in range(n_aliens)])
    # Intrusion errors
    make_plot(hum_summary_initial_learn[IL_cols[3:5]], ag_summary[IL_cols[3:5]],
              hum_summary_initial_learn[IL_cols[6:8]].values, ag_summary[IL_cols[6:8]].values,
              ylabel="Intrusion errors", hline=1/3, ylim=(0, 0.5), xticklabels=['Accuracy', 'Intrusion (prev.)'])
    # TS values affect performance
    make_plot(hum_summary_initial_learn[IL_cols[10:12]], ag_summary[IL_cols[10:12]],
              hum_summary_initial_learn[IL_cols[13:15]].values, ag_summary[IL_cols[13:15]].values,
              ylabel="TS performance", hline=1/3, ylim=(1/3, 0.65), xticklabels=['TS5', 'TS4'])
    # TS values affect preference
    make_plot(hum_summary_competition[CO_cols[:2]], ag_summary[CO_cols[:2]],
              hum_summary_competition[CO_cols[2:4]].values,
              ylabel="% better chosen", hline=1/2, ylim=(0, 0.65), xticklabels=['Context', 'Context-stimulus'])
    # TS values affect generalization
    make_plot(hum_summary_rainbow[RB_sum_cols[:4]], ag_summary_rainbow[RB_sum_cols[:4]],
              hum_summary_rainbow[RB_sum_cols[4:8]].values,
              ylabel='% TS selected', ylim=(0, 0.45), xticklabels=['TS6', 'TS5', 'TS4', 'None'])
    # Savings
    make_plot(hum_summary_initial_learn[IL_cols[:3]], ag_summary[IL_cols[:3]],
              ylabel="Savings")

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    [ax.set_title(title) for ax, title in zip(axes, ['Humans', 'Simulations'])]
    axes[0].matshow(hum_rainbow_dat[0])
    axes[1].matshow(ag_rainbow_dat.astype(float))
    [ax.set_xlabel('Aliens') for ax in axes]
    [ax.set_ylabel('Actions') for ax in axes]

    # # Show values of correct actions in rainbow phase
    # correct_TS = task.TS.copy().astype(float)
    # correct_TS[correct_TS == 1] = np.nan
    # av_Q_correct_action = np.nanmean(correct_TS, axis=0)
    # av_Q_correct_action[np.isnan(av_Q_correct_action)] = 0
    # axes[1].set_title("Values correct actions")
    # axes[1].matshow(av_Q_correct_action)
    # for (i, j), z in np.ndenumerate(av_Q_correct_action):
    #     axes[1].text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
    # axes[0].set_xlabel('Aliens')
    # axes[0].set_ylabel('Actions')


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
    # TS_choices = get_summary_rainbow(n_aliens, n_seasons, rainbow_dat, task)
    # summary_rainbow = np.mean(TS_choices, axis=1)

    return list(parameters.values) + summary_initial_learn + list(summary_cloudy) + list(summary_competition) + list(rainbow_dat.flatten())


# Get summaries for different parameters
if do_calculate_summaries:
    summaries = pd.DataFrame(np.full((n_iter, len(summary_dat_cols)), np.nan), columns=summary_dat_cols)
    for iter in range(n_iter):
        print("Iteration {}".format(iter))

        params = np.random.rand(len(param_names))
        # params = [0.01, 6.5/20, 0.005, 0.3, 4.5/20, 0.005]  # TODO: debug - remove!
        params = selected_agents.loc[iter, param_names]
        params[['beta', 'beta_high']] = (params[['beta', 'beta_high']] - param_ranges.loc[0, ['beta', 'beta_high']]) / (param_ranges.loc[1, ['beta', 'beta_high']] - param_ranges.loc[0, ['beta', 'beta_high']])  # rescale beta to 0-1
        # params = [0.1, 0.8, 0]
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
    filenames = [filename for filename in filenames if 'alpha' in filename]  # don't read in selected_agents.csv etc.
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
    all_summaries['IL_perf_TS2minus1'] = all_summaries['IL_perf_TS2'] - all_summaries['IL_perf_TS1']
    all_summaries['CO_season_minus_alien'] = all_summaries['CO_acc_season'] - all_summaries['CO_acc_season_alien']

    # Plot correlations and histograms
    for model_name in models:
        model_summaries = all_summaries.loc[all_summaries['model'] == model_name]
        model_summaries = model_summaries.reset_index(drop=True)
        pd.plotting.scatter_matrix(model_summaries.loc[:1000, ['alpha', 'beta', 'forget']])

    # Plot savings (Initial Learning)
    plt.figure()
    for i, effect in enumerate(IL_cols[:3] + ['IL_saving_last_minus_first']):
        plt.subplot(2, 2, i+1)
        # bins = np.arange(min(all_summaries[effect]), max(all_summaries[effect]), 0.01)
        plt.axvline(x=0, color='grey', linestyle='--')
        plt.axvline(x=hum_summary_initial_learn[effect].values, color='red', linestyle='-')
        plt.ylim(0, 25)
        plt.xlim(-0.1, 0.5)
        for model in models:
            dat = all_summaries.loc[all_summaries['model'] == model]
            # plt.hist(dat[effect], bins=bins, alpha=0.5, label=model, density=True)
            sns.distplot(dat[effect], kde=True, hist=True, label=model)
        plt.xlabel(effect)
        plt.ylabel("Probability density")
    plt.legend()
    plt.tight_layout()

    # def make_histograms(sim_dat, hum_dat, ylabel="", xlabel="", xlim=False, vline=False):
    #     fig, axes = plt.subplots(ncols=2, figsize=(6, 3))
    #     [ax.axvline(x=1 / 3, color='grey', linestyle='--') for ax in axes]
    #     [ax.set_xlim(0, 1) for ax in axes]
    #     [ax.set_ylabel("Probability density") for ax in axes]
    #     # [ax.set_yscale('log') for ax in axes]
    #     for i, effect in enumerate(IL_cols[3:5]):
    #         axes[i].axvline(x=hum_summary_initial_learn[effect].values, color='red', linestyle='-')
    #         for model in models:
    #             dat = all_summaries.loc[all_summaries['model'] == model]
    #             sns.distplot(dat[effect], kde=True, hist=True, label=model, ax=axes[i])
    #         axes[i].set_xlabel(effect)
    #     plt.legend()
    #     plt.tight_layout()

    # Plot intrusion errors (Initial Learning)
    fig, axes = plt.subplots(ncols=2, figsize=(6, 3))
    [ax.axvline(x=1 / 3, color='grey', linestyle='--') for ax in axes]
    [ax.set_xlim(0, 1) for ax in axes]
    [ax.set_ylabel("Probability density") for ax in axes]
    # [ax.set_yscale('log') for ax in axes]
    for i, effect in enumerate(IL_cols[3:5]):
        axes[i].axvline(x=hum_summary_initial_learn[effect].values, color='red', linestyle='-')
        for model in models:
            dat = all_summaries.loc[all_summaries['model'] == model]
            sns.distplot(dat[effect], kde=True, hist=True, label=model, ax=axes[i])
        axes[i].set_xlabel(effect)
    plt.legend()
    plt.tight_layout()

    # Plot TS values (Initial Learning)
    plt.figure()
    for i, effect in enumerate(IL_cols[6:9]):
        plt.subplot(2, 2, i+1)
        # bins = np.arange(min(all_summaries[effect]), max(all_summaries[effect]), 0.01)
        plt.axvline(x=hum_summary_initial_learn[effect].values, color='red', linestyle='-')
        plt.xlim(0, 1)
        # plt.ylim(0, 10)
        for model in models:
            dat = all_summaries.loc[all_summaries['model'] == model]
            # plt.hist(dat[effect], bins=bins, alpha=0.5, label=model, density=True)
            sns.distplot(dat[effect], kde=True, hist=True, label=model)
        plt.xlabel(effect)
        plt.ylabel("Probability density")

    plt.subplot(2, 2, 4)
    effect = 'IL_perf_TS2minus1'  # 'IL_perf_TS_corr'
    plt.axvline(x=hum_summary_initial_learn[effect].values, color='red', linestyle='-')
    # bins = np.arange(min(all_summaries[effect]), max(all_summaries[effect]), 0.01)
    for model in models:
        dat = all_summaries.loc[all_summaries['model'] == model]
        # plt.hist(dat[effect], bins=bins, alpha=0.5, label=model, density=True)
        sns.distplot(dat[effect], kde=True, hist=True, label=model)
    plt.xlabel("Slope (TS2-TS1)")  # "Correlation (r)")
    plt.ylabel("Probability density")
    plt.legend()
    plt.tight_layout()

    # Plot TS reactivation (Cloudy Season)
    plt.figure()
    for i, effect in enumerate(CL_cols[4:]):
        plt.subplot(2, 2, i+1)
        # bins = np.arange(min(all_summaries[effect]), max(all_summaries[effect]), 0.02)
        plt.axvline(x=0, color='grey', linestyle='--')
        plt.axvline(x=hum_summary_cloudy[effect].values, color='red', linestyle='-')
        plt.ylim(0, 0.5)  # TODO: comment in to get zoom-in version for paper
        for model in models:
            dat = all_summaries.loc[all_summaries['model'] == model]
            # plt.hist(dat[effect], bins=bins, alpha=0.5, label=model, density=True)
            sns.distplot(dat[effect], kde=True, hist=True, label=model)
        plt.xlabel(effect)
        plt.ylabel("Probability density")
    plt.legend()
    plt.tight_layout()

    # Competition phase
    plt.figure()
    for i, effect in enumerate(CO_cols[:2]):
        plt.subplot(3, 1, i+1)
        # bins = np.arange(min(all_summaries[effect]), max(all_summaries[effect]), 0.01)
        plt.axvline(x=1/2, color='grey', linestyle='--')
        plt.axvline(x=hum_summary_competition[effect].values, color='red', linestyle='-')
        plt.xlim(0, 1)
        for model in models:
            dat = all_summaries.loc[all_summaries['model'] == model]
            # plt.hist(dat[effect], bins=bins, alpha=0.5, label=model, density=True)
            sns.distplot(dat[effect], kde=True, hist=True, label=model)
        plt.xlabel("% selected better {}".format(effect[7:]))
        plt.ylabel("Probability density")

    effect = 'CO_season_minus_alien'
    plt.subplot(3, 1, 3)
    # bins = np.arange(min(all_summaries[effect]), max(all_summaries[effect]), 0.01)
    plt.axvline(x=0, color='grey', linestyle='--')
    plt.axvline(x=hum_summary_competition[effect].values, color='red', linestyle='-')
    plt.xlim(-0.2, 0.2)
    for model in models:
        dat = all_summaries.loc[all_summaries['model'] == model]
        # plt.hist(dat[effect], bins=bins, alpha=0.5, label=model, density=True)
        sns.distplot(dat[effect], kde=True, hist=True, label=model)
    plt.xlabel("Season minus season-alien")
    plt.ylabel("Probability density")
    plt.legend()
    plt.tight_layout()

    # Rainbow phase TS choices
    plt.figure()
    for model in models:

        # Get summary_rainbow
        rainbow_dat = all_summaries.loc[all_summaries['model'] == model, RB_cols]
        rainbow_dat = rainbow_dat.values.reshape((rainbow_dat.shape[0], n_aliens, n_actions))
        summary_rainbow = np.array([get_summary_rainbow(n_aliens, n_seasons, dat, task) for dat in rainbow_dat])
        summary_rainbow = pd.DataFrame(summary_rainbow, columns=RB_sum_cols)

        # Plot
        for i, effect in enumerate(RB_sum_cols[:3]):
            plt.subplot(2, 3, i+1)
            sns.distplot(summary_rainbow[effect], kde=True, hist=True, label=model)
            plt.axvline(x=10/12/3, color='grey', linestyle='--')
            plt.axvline(x=hum_summary_rainbow[effect].values, color='red', linestyle='-')
            plt.xlim(0, 0.5)
            plt.ylim(0, 60)
            plt.xlabel(effect)
            plt.ylabel("Probability density")

        effect = RB_sum_cols[3]
        plt.subplot(2, 3, 4)
        sns.distplot(summary_rainbow[effect], kde=True, hist=True, label=model)
        plt.axvline(x=(2/12), color='grey', linestyle='--')
        plt.axvline(x=hum_summary_rainbow[effect].values, color='red', linestyle='-')
        plt.xlim(0, 0.5)
        plt.ylim(0, 60)
        plt.xlabel(effect)
        plt.ylabel("Probability density")

        effect = 'TS2minusTS0'
        plt.subplot(2, 3, 5)
        sns.distplot(summary_rainbow[effect], kde=True, hist=True, label=model)
        plt.axvline(x=0, color='grey', linestyle='--')
        plt.axvline(x=hum_summary_rainbow[effect].values, color='red', linestyle='-')
        plt.xlim(-0.3, 0.3)
        plt.ylim(0, 60)
        plt.xlabel(effect)
        plt.ylabel("Probability density")

    plt.legend()
    plt.tight_layout()

    # Rainbow phase TS choices
    plt.figure()
    colors = sns.cubehelix_palette(3, start=.5, rot=-.75, reverse=True)
    for i, model in enumerate(models):
        ax1 = plt.subplot(2, 2, i + 1)

        # Get summary_rainbow
        rainbow_dat = all_summaries.loc[all_summaries['model'] == model, RB_cols]
        rainbow_dat = rainbow_dat.values.reshape((rainbow_dat.shape[0], n_aliens, n_actions))
        summary_rainbow = np.array([get_summary_rainbow(n_aliens, n_seasons, dat, task) for dat in rainbow_dat])
        summary_rainbow = pd.DataFrame(summary_rainbow, columns=RB_sum_cols)

        # Plot
        for j, effect in enumerate(RB_sum_cols[:3]):
            sns.distplot(summary_rainbow[effect], kde=True, hist=True, label=effect, color=colors[j], ax=ax1)
            plt.axvline(x=10/12/3, color='grey', linestyle='--')
            # plt.axvline(x=hum_summary_rainbow[effect].values, color=colors[j], linestyle='-')
            plt.xlim(0, 0.5)
            plt.ylim(0, 40)
            plt.xlabel(model)
            plt.ylabel("Probability density")
        plt.legend()

        effect = 'TS2minusTS0'
        ax2 = plt.subplot(2, 2, 3)
        sns.distplot(summary_rainbow[effect], kde=True, hist=True, label=model, ax=ax2)
        plt.axvline(x=0, color='grey', linestyle='--')
        plt.axvline(x=hum_summary_rainbow[effect].values, color='red', linestyle='-')
        plt.xlim(-0.3, 0.3)
        plt.ylim(0, 40)
        plt.xlabel(effect)
        plt.ylabel("Probability density")
        plt.legend()
    plt.tight_layout()

    # Find a good subset of parameters - define criteria
    IL_acc_prev_TS = (
            all_summaries['IL_acc_prev_TS'] > 3/4 * hum_summary_initial_learn['IL_acc_prev_TS'].values[0]) & (
            all_summaries['IL_acc_prev_TS'] < 3/2 * hum_summary_initial_learn['IL_acc_prev_TS'].values[0])
    IL_perf_TS2minus1 = (
            all_summaries['IL_perf_TS2minus1'] < 3/4 * hum_summary_initial_learn['IL_perf_TS2minus1'].values[0]) & (
            all_summaries['IL_perf_TS2minus1'] > 3/2 * hum_summary_initial_learn['IL_perf_TS2minus1'].values[0])
    CL_slope = (
            all_summaries['CL_slope'] > 3/4 * hum_summary_cloudy['CL_slope'].values[0]) & (
            all_summaries['CL_slope'] < 3/2 * hum_summary_cloudy['CL_slope'].values[0])
    CO_season_minus_alien = (
           all_summaries['CO_season_minus_alien'] > 3/4 * hum_summary_competition['CO_season_minus_alien'].values[0]) & (
           all_summaries['CO_season_minus_alien'] < 3/2 * hum_summary_competition['CO_season_minus_alien'].values[0])
    # RB_TS2minusTS0 = rainbow_dat['TS2minutsTS0']

    # Subset data and plot
    selected_agents = all_summaries.loc[
        IL_acc_prev_TS & IL_perf_TS2minus1 & CL_slope & CO_season_minus_alien]
    # Save selected_agents as csv
    if do_save_selected_agents:
        save_path = plot_dir + 'selected_agents.csv'
        print("Saving selected_agents to {}".format(save_path))
        selected_agents.to_csv(save_path, index=False)

    selected_agents[['beta', 'beta_high']] /= 20
    selected_agents[param_names].plot(kind='bar')
    selected_agents[param_names].plot(kind='box', by='model')
    scatter_matrix(selected_agents[param_names])
    plt.show()

    # Intrusion error heatmap (prev_TS)
    fig = plt.figure(figsize=(10, 5))
    for i, effect in enumerate(['IL_acc_prev_TS', 'IL_acc_current_TS']):

        # Subset data with human-like behavior
        dat = all_summaries.loc[(all_summaries['model'] == 'hier') &
                                (all_summaries['IL_acc_prev_TS'] > 0.55)]

        # Plot raw parameters
        ax = fig.add_subplot(2, 3, (i*3)+1, projection='3d')
        ax.set_title(effect)
        ax.scatter(dat['alpha'], dat['beta'], dat['forget'], c=dat[effect], label=dat[effect], marker='.')
        ax.set_xlabel('alpha')
        ax.set_ylabel('beta')
        ax.set_zlabel('forget')

        ax = fig.add_subplot(2, 3, (i*3)+2, projection='3d')
        ax.set_title(effect)
        ax.scatter(dat['alpha_high'], dat['beta_high'], dat['forget_high'], c=dat[effect], label=dat[effect], marker='.')
        ax.set_xlabel('alpha_high')
        ax.set_ylabel('beta_high')
        ax.set_zlabel('forget_high')

        if do_isomap:
            # Dimensionality reduction
            n_neighbors = 12
            n_components = 3
            params_norm = preprocessing.StandardScaler().fit_transform(dat[param_names])  # standardize data
            Y = manifold.Isomap(n_neighbors, n_components).fit_transform(params_norm)  # apply Isomap

            # Plot dimensionality-reduced parameters
            ax = fig.add_subplot(2, 3, (i*3)+3, projection='3d')
            ax.set_title("Isomap " + effect)
            ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=dat[effect], marker='.')

        plt.tight_layout()

    # # Calculate Isomap & Co. and visualize
    # params_norm = preprocessing.StandardScaler().fit_transform(dat[param_names])  # standardize data
    # effect = dat['IL_acc_prev_TS']
    # fig = plt.figure(figsize=(15, 5))
    #
    # i = 1
    # for n_components in range(2, 6, 2):
    #     for n_neighbors in range(2, 20, 3):
    #
    #         # Reduce dimensionality
    #         # Y = decomposition.PCA(n_components).fit_transform(params_norm)
    #         # mds = manifold.MDS(n_neighbors, max_iter=100, n_init=1)
    #         # Y = mds.fit_transform(params_norm)
    #         # Y = manifold.LocallyLinearEmbedding(n_neighbors, n_components, eigen_solver='auto',
    #         #                                     method='standard').fit_transform(params_norm)
    #         # Y = manifold.LocallyLinearEmbedding(n_neighbors, n_components, eigen_solver='auto',  # n_neighbors >= 6
    #         #                                     method='modified').fit_transform(params_norm)
    #         Y = manifold.Isomap(n_neighbors, n_components).fit_transform(params_norm)
    #
    #         # Plot
    #         ax = fig.add_subplot(2, 6, i)
    #         plt.scatter(Y[:, 0], Y[:, 1], c=effect)
    #         plt.title('{0} comp., {1} neigh.'.format(n_components, n_neighbors))
    #         i += 1
    # plt.tight_layout()
    plt.show()

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
