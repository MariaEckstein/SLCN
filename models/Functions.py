import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RepeatedKFold


# def replace_nans(data):
#
#     data[np.isnan(data)] = np.random.binomial(1, 0.5, np.sum(np.isnan(data)))
#     return data

# # Example use
# replace_nans(np.full(10, np.nan))

# def load_real_mouse_data(data_dir='C:/Users/maria/MEGAsync/SLCN/PSMouseData/', n_trials=782):
#     """
#     :param data_dir: Where is the data stored?
#     :param exclusion_percentile_ntrials: What percentage of mouse-sessions are allowed to have fewer trials than others?
#     When it is 0, all sessions will be shortened to the shorted occurence; when it is 1, all data will be kept, and
#     shorter ones will be filled up with np.nan.
#     :return:
#     """
#
#     # Load mouse data
#     rewards_j = pd.read_csv(os.path.join(data_dir, 'Juvi_Reward.csv')).T.values  # after reading in: [trials x animals]
#     rewards_a = pd.read_csv(os.path.join(data_dir, 'Adult_Reward.csv')).T.values
#     rewards = np.hstack([rewards_j, rewards_a])
#
#     actions_j = pd.read_csv(os.path.join(data_dir, 'Juvi_Choice.csv')).T.values
#     actions_a = pd.read_csv(os.path.join(data_dir, 'Adult_Choice.csv')).T.values
#     actions = np.hstack([actions_j, actions_a])
#
#     correct_actions_j = pd.read_csv(os.path.join(data_dir, 'Juvi_TaskData.csv')).T.values
#     correct_actions_a = pd.read_csv(os.path.join(data_dir, 'Adult_TaskData.csv')).T.values
#     correct_actions = np.hstack([correct_actions_j, correct_actions_a])  # Which action was the correct one on each trial?
#
#     corrects = (actions == correct_actions).astype('int')  # When did mice choose the right action?
#
#     fullID_j = pd.read_csv(os.path.join(data_dir, 'Juvi_AnimalID.csv')).T.values.flatten()
#     fullID_a = pd.read_csv(os.path.join(data_dir, 'Adult_AnimalID.csv')).T.values.flatten()
#     fullIDs = np.concatenate([fullID_j, fullID_a])
#
#     # Determine number of trials
#     n_trials_per_animal = np.sum(np.invert(np.isnan(actions)), axis=0)
#     sns.distplot(n_trials_per_animal[:len(fullID_j)], label='juv')
#     sns.distplot(n_trials_per_animal[len(fullID_j):], label='ad')
#     plt.legend()
#
#     # if type(exclusion_percentile_ntrials) == int:
#     #     n_trials = np.round(np.percentile(n_trials_per_animal, exclusion_percentile_ntrials)).astype('int')
#     # elif n_trials:
#     #     pass
#     # else:
#     #     raise(ValueError, 'You must either provide "exclusion_percentile_ntrials" or "n_trials" to this function.')
#
#     # Remove interleaved na trials by shifting up later trials
#     missed_trials = np.isnan(actions)
#     rewards[missed_trials] = np.nan
#
#     rewards = pd.DataFrame(rewards)
#     rewards = rewards.apply(lambda x: pd.Series(x.dropna().values))
#
#     corrects = pd.DataFrame(corrects)
#     corrects = corrects.apply(lambda x: pd.Series(x.dropna().values))
#
#     correct_actions = pd.DataFrame(correct_actions)
#     correct_actions = correct_actions.apply(lambda x: pd.Series(x.dropna().values))
#
#     actions = pd.DataFrame(actions)
#     actions = actions.apply(lambda x: pd.Series(x.dropna().values))
#
#     # Cut at n_trials
#     rewards = rewards[:n_trials]
#     actions = actions[:n_trials]
#     corrects = corrects[:n_trials]
#     correct_actions = correct_actions[:n_trials]
#
#     assert np.shape(rewards) == np.shape(actions)
#
#     return {
#         'rewards': rewards,
#         'actions': actions,
#         'corrects': corrects,
#         'correct_actions': correct_actions,
#         'fullIDs': fullIDs,
#         'n_trials_per_animal': n_trials_per_animal,
#         # 'n_trials': n_trials
#     }


# def load_one_measure(name, data_dir):
#     measure_j = pd.read_csv(
#         os.path.join(data_dir, 'Juvi_{}.csv'.format(name))).T.values  # after reading in: [trials x animals]
#     measure_a = pd.read_csv(os.path.join(data_dir, 'Adult_{}.csv'.format(name))).T.values
#     measure_dat = np.hstack([measure_j, measure_a])
#
#     return measure_dat
#
# # # Example use
# # load_one_measure('Reward', mouse_data_dir)


# def remove_na_trials(measure_dat, missed_trials):
#     measure_dat = pd.DataFrame(measure_dat)
#     measure_dat[missed_trials] = np.nan
#     measure_dat = measure_dat.apply(lambda x: pd.Series(x.dropna().values))
#
#     return measure_dat

# # Example use
# remove_na_trials(load_one_measure('Reward', mouse_data_dir))


# def load_mouse_data(data_dir):
#
#     # Load mouse data
#     rewards = load_one_measure('Reward', data_dir)
#     rts = load_one_measure('ITI', data_dir)  # Lung-Hao: ITI is the time of last nose poke event (in or out) of previous trial to center poke of current trial. So it's the ITI proceeding the current trial. The first trial has ITI because we removed the trials before first switch.
#     actions = load_one_measure('Choice', data_dir)
#     correct_actions = load_one_measure('TaskData', data_dir)
#     corrects = (actions == correct_actions).astype('int')  # When did mice choose the right action?
#
#     fullID_j = pd.read_csv(os.path.join(data_dir, 'Juvi_AnimalID.csv')).T.values.flatten()
#     fullID_a = pd.read_csv(os.path.join(data_dir, 'Adult_AnimalID.csv')).T.values.flatten()
#     fullIDs = np.concatenate([fullID_j, fullID_a])
#
#     # Remove interleaved na trials by shifting up later trials
#     missed_trials = np.isnan(actions)
#     actions = remove_na_trials(actions, missed_trials)
#     rewards = remove_na_trials(rewards, missed_trials)
#     rts = remove_na_trials(rts, missed_trials)
#     corrects = remove_na_trials(corrects, missed_trials)
#     correct_actions = remove_na_trials(correct_actions, missed_trials)
#
#     # Make sure all dataframes have the same shape
#     assert np.shape(rewards) == np.shape(actions)
#     assert np.shape(corrects) == np.shape(correct_actions)
#     assert np.shape(rewards) == np.shape(correct_actions)
#
#     return {
#         'actions': actions,
#         'rewards': rewards,
#         'corrects': corrects,
#         'rts': rts,
#         'correct_actions': correct_actions,
#         'fullIDs': fullIDs,
#     }
#
# # # Example use
# # raw_dat = load_mouse_data(mouse_data_dir)
# # raw_dat


def get_info_from_fullID(fullID, column_name):
    """formula: session_ID=[session_ID;animal_idn*100000 + age*100 + (strcmp(animal_gender,'F')+1)*10 + strcmp(animal_treatment,'Juvenile')+1];
    """

    if column_name == 'agegroup':
        a = int(str(fullID)[-1:])
        if a == 1:
            return 'Adult'
        elif a == 2:
            return 'Juvenile'
        else:
            raise ValueError('Invalid fullID at agegroup.')

    elif column_name == 'sex':
        g = int(str(fullID)[-2:-1])
        if g == 1:
            return 'Male'
        elif g == 2:
            return 'Female'

    elif column_name == 'age':
        return int(str(fullID)[-5:-2])

    elif column_name == 'animal':
        return int(str(fullID)[:-5])

# # Example use
# get_info_from_fullID(fullIDs[0], 'agegroup')


def add_meta_column(data):

    meta = np.zeros(data.shape[0]).astype(int).astype(str)
    meta[data.session <= 2] = '1-3'
    meta[(data.session >= 3) & (data.session <= 7)] = '4-8'
    meta[(data.session >= 8) & (data.session <= 10)] = '9-11'
    meta[meta == '0'] = np.nan

    return meta

# # Example use
# add_meta_column(true_dat)


def get_session(ani_dat):

    return np.append([0], np.cumsum((np.diff(ani_dat.age)).astype(bool).astype(int)))

# # Example use
# ani_dat = true_dat.loc[true_dat.animal == 14]
# sessions = get_session(ani_dat)
# print(np.unique(sessions), len(sessions), ani_dat.shape[0])


# MetaSLCN
def sigmoid(vector):
    return 1 / (1 + np.exp(-vector))


# # Example
# x = np.arange(-30, 30, 0.1)
# dat = pd.DataFrame({'x': x, 'y': 3 * sigmoid(x / 3)})
# (gg.ggplot(dat, gg.aes('x', 'y')) +
#  gg.geom_point()
#  )

def zscore(vector):
    return (vector - vector.mean()) / vector.std()


# # Example
# x = np.arange(-30, 0, 0.1)
# dat = pd.DataFrame({'x': x, 'y': zscore(x)})
# (gg.ggplot(dat, gg.aes('x', 'y')) +
#  gg.geom_point()
#  )

# param_names = ['alpha', 'beta', 'persev', 'forget', 'epsilon', 'wm', 'p_switch_inv', 'p_reward', 'K']


def name_from_index(index, names):
    for name in names:
        if name in index:
            return name


def get_ages_cols():
    # return ['ID', 'PreciseYrs', 'BMI', 'Gender', 'meanT', 'PDS']
    return ['ID', 'PreciseYrs', 'BMI', 'meanT', 'PDS', 'sex', 'age_group_', 'PDS_group_', 'T_group_']


# def get_category_cols():
#     return ['minus', 'delta', 'stay', 'ACC', 'RT', 'lrn', 'miss', 'WS', 'LS']

def get_category_dict():

    tasks = ['rl', 'bf', 'ps']
    return {

        # Performance
        'miss': ['{}_miss'.format(t) for t in tasks],
        'RT': ['{}_RT'.format(t) for t in tasks],
        'RTsd': ['{}_RTsd'.format(t) for t in tasks],
        'ACC': ['{}_ACC'.format(t) for t in tasks],
        'ACC2': ['bf_ACC_first3trials', 'bf_asymptote', 'bf_intercept', 'ps_n_switches', 'ps_criterion_trial',],

        # Stay
        'Stay': ['rl_stay_choice', 'bf_stay_choice', 'ps_stay', 'rl_stay_motor', 'bf_stay_motor'],
        'Persev': ['ps_persev'],

        # WSLS (win-stay & lose-STAY)
        'WS': ['{}_WS'.format(t) for t in tasks],
        'WS2': ['ps_WLS', 'bf_prew'],
        'LS': ['{}_LS'.format(t) for t in tasks],

        # Forgetting / effects of time
        'Delay': ['bf_delay', 'rl_lrn_delay_sig'],
        'Forget': ['bf_forget', 'rl_forget'],

        # Interesting parameters
        'Beta': ['rl_epsilon_1_over_beta', 'bf_1_over_beta', 'ps_1_over_beta'],
        'Alpha': ['rl_log_alpha', 'bf_alpha', 'ps_alpha'],
        'Nalpha': ['rl_log_nalpha', 'ps_nalpha'],

        # Other parameters
        'Bayes': ['ps_p_switch', 'ps_p_reward'],

        # WM stuff
        'WM': ['rl_rho', 'rl_K', 'rl_lrn_ns_sig', 'rl_ACC_ns_slope', 'rl_RT_ns_slope'],

        # Learning / performance improvement
        'Learn':
            # ['{}_ACC_delta'.format(t) for t in tasks]
            # + ['{}_RT_delta'.format(t) for t in tasks]
            # + ['rl_RT_ns_slope_delta', 'rl_ACC_ns_slope_delta', ]
            ['rl_lrn_pcor_sig', 'rl_lrn_pinc_sig', 'bf_learning_slope', 'ps_ACC_delta'],
    }
# # Example use
# category_dict = get_category_dict()
# category_dict


def get_key(my_dict, val):
    for key, value in my_dict.items():
        if val in value:
            return key
# # Example use
# get_key(category_dict, 'ps_RT')


def get_category_from_feature(feature_col, category_dict):

    # Initialize category
    category = feature_col.copy()

    # List measures for which I have categories
    dict_measures = []
    for measures in category_dict.values():
        dict_measures += measures

    # Retrieve the categories of each measure
    for meas in dict_measures:
        # category[[meas in c for c in category]] = get_key(category_dict, meas)
        category[meas == category] = get_key(category_dict, meas)

    return category
# # Example use
# get_category_from_feature(all_data_long, category_dict)


def get_param_names():
    return ['alpha', 'beta', 'persev', 'forget', 'epsilon', 'rho', 'p_switch', 'p_reward', 'K']


def bool_from_index(index, names):
    contains_name = False
    for param_name in names:
        if param_name in index:
            contains_name = True

    return contains_name


def get_quantile_groups(data, col, quantiles=np.array([0, 1/4, 2/4, 3/4])):

    quant_col = np.repeat(np.nan, data.shape[0])

    # Adult quantiles
    quant_col[(data.PreciseYrs >= 18) & (data.ID >= 400)] = 2  # UCB
    quant_col[(data.PreciseYrs >= 18) & (data.ID >= 300) & (data.ID < 400)] = 3  # Community adults

    # Determine teen quantiles, separately for both genders
    cut_off_values_both = {}
    for gender in [1, 2]:

        # Get 4 quantiles
        cut_off_values = np.nanquantile(
            data.loc[(data.PreciseYrs < 18) & (data.Gender == gender), col], quantiles)
        cut_off_values_both[gender] = cut_off_values

        # Name 4 quantiles
        for cut_off_value, quantile in zip(cut_off_values, np.round(quantiles + 1 / len(quantiles), 2)):
            quant_col[(data.PreciseYrs < 18) & (data.Gender == gender) & (data[col] >= cut_off_value)] = quantile

    return quant_col, cut_off_values_both


# # Example use
# ages['age_group'], cut_off_values = get_quantile_groups(ages, 'PreciseYrs')
# cut_off_values, ages.loc[range(17, 70), ['PreciseYrs', 'Gender', 'age_group']]


class RepeatableGridSearchCV:
    """ Wrapper class for GridSearchCV with RepeatedKFold.
    """

    def __init__(
            self,
            model,
            parameters,
            n_splits=20,
            n_repeats=10,
            random_state=None,
            n_jobs=-1,
            scoring=None,
            refit=True,
    ):

        # cv or repeated cv?
        if n_repeats == 1:
            cv = n_splits
        else:
            cv = RepeatedKFold(
                n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
            )

        # setup the grid search; done
        gridder = GridSearchCV(
            model, parameters, cv=cv, n_jobs=n_jobs, scoring=scoring, refit=refit
        )
        self._gridder = gridder

    def __getattr__(self, attr):

        """ Expose all attributes from GridSearchCV """

        return getattr(self._gridder, attr)


def make_gridder_pd(gridder):

    return pd.DataFrame({
        name: gridder.cv_results_[name]
        for name in ['mean_test_score', 'std_test_score', 'param_alpha']  # , 'mean_train_score', 'std_train_score']
    })

# # Example use
# make_gridder_pd(gridder)