import numpy as np
import os


def check_user_settings(settings_dictionary):

    # Check if all user-specified settings are allowed; raise error if not
    sets = settings_dictionary
    allowed_values = {'run_on_cluster': [True, False],
                      'data_set': ['PS', 'Aliens'],
                      'learning_style': ['s-flat', 'flat', 'hierarchical', 'Bayes'],
                      'use_humans': [True, False],
                      'set_specific_parameters': [True, False]}
    for parameter in allowed_values.keys():
        assert sets[parameter] in allowed_values[parameter], 'Variable "{0}" must be one of {1}.' \
            .format(parameter, allowed_values[parameter])


def get_paths(sets):

    # Get paths on cluster
    paths = dict()
    if sets['run_on_cluster']:
        paths['base_path'] = '/home/bunge/maria/Desktop/' + sets['data_set']
        paths['human_data_path'] = paths['base_path'] + '/humanData/'

    # Get paths on local computer
    else:
        if sets['data_set'] == 'Aliens':
            paths['base_path'] = 'C:/Users/maria/MEGAsync/Berkeley/TaskSets'
            paths['human_data_path'] = paths['base_path'] + '/Data/version3.1/'
        else:
            paths['base_path'] = 'C:/Users/maria/MEGAsync/SLCN'
            paths['human_data_path'] = paths['base_path'] + 'data/PSResults'

    # Define output folders
    paths['agent_data_path'] = paths['base_path'] + '/' + sets['data_set'] + 'GenRec/'

    if sets['use_humans']:
        paths['plot_data_path'] = paths['human_data_path'] + '/heatmap_data/'
        paths['fitting_save_path'] = paths['human_data_path'] + '/fit_pars_Bayes/'
        paths['file_name_pattern'] = sets['data_set'] + '_'
    else:
        paths['plot_data_path'] = paths['agent_data_path'] + '/heatmap_data/'
        paths['fitting_save_path'] = paths['agent_data_path'] + '/fit_pars/'
        paths['file_name_pattern'] = 'sim_'
    paths['plot_save_path'] = paths['plot_data_path'] + '/plots/'

    # Create output folders
    if not os.path.isdir(paths['agent_data_path']):
        os.makedirs(paths['agent_data_path'])
    if not os.path.isdir(paths['plot_data_path']):
        os.makedirs(paths['plot_data_path'])
    if not os.path.isdir(paths['plot_save_path']):
        os.makedirs(paths['plot_save_path'])
    if not os.path.isdir(paths['fitting_save_path']):
        os.makedirs(paths['fitting_save_path'])

    return paths


def get_agent_stuff(data_set, learning_style, fit_par_names):

    if data_set == 'Aliens':
        n_TS = 3
        beta_scaler = 2
        beta_high_scaler = 4
        TS_bias_scaler = 5
    else:
        n_TS = 2
        beta_scaler = 10
        beta_high_scaler = 10
        TS_bias_scaler = np.nan

    return {'name': data_set,
            'learning_style': learning_style,
            'id': 0,
            'n_TS': n_TS,
            'fit_par': '_'.join(fit_par_names),
            'beta_scaler': beta_scaler,
            'beta_high_scaler': beta_high_scaler,
            'TS_bias_scaler': TS_bias_scaler}


def get_task_stuff(data_set):

    if data_set == 'PS':
        task_stuff = {'n_actions': 2,
                      'p_reward': 0.75,
                      'n_trials': 200,
                      'path': 'C:/Users/maria/MEGAsync/SLCN/ProbabilisticSwitching/Prerandomized sequences',
                      'av_run_length': 10}  # DOUBLE-CHECK!!!!

    else:
                        # TS0
        TSs = np.array([[[1, 6, 1],  # alien0, items0-2
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

        task_stuff = {'phases': ['1InitialLearning', '2CloudySeason', 'Refresher2', '3PickAliens',
                                 'Refresher3', '5RainbowSeason', 'Mixed'],
                      'n_trials_per_alien': np.array([13, 10, 7, np.nan, 7, 1, 7]),
                      'n_blocks': np.array([3, 3, 2, np.nan, 2, 3, 3]),
                      'n_aliens': 4,
                      'n_actions': 3,
                      'n_contexts': 3,
                      'TS': TSs}

    return task_stuff


def get_comp_stuff(data_set):
    if data_set == 'Aliens':
        return {'phases': ['season', 'alien-same-season', 'item', 'alien'],
                'n_blocks': {'season': 3, 'alien-same-season': 3, 'item': 3, 'alien': 3}}
    else:
        return None


def get_parameter_stuff(data_set, fit_par_names):

    par_names = ['alpha', 'alpha_high', 'beta', 'beta_high', 'epsilon', 'forget', 'TS_bias']
    default_pars = np.array([.1,  99,       .5,       99,       0.,        0.,       .3])
    par_hard_limits = ((0., 1.), (0., 1.), (0., 1.), (0., 1.), (0., 1.),  (0., 1.), (0., 1.))
    par_soft_limits = ((0., .5), (0., .5), (0., 1.), (0., 1.), (0., .25), (0., .1), (0., 1.))

    # For PS agent, remove TS_bias parameter
    if data_set == 'PS':
        par_names = par_names[:-1]
        default_pars = default_pars[:-1]
        par_hard_limits = par_hard_limits[:-1]
        par_soft_limits = par_soft_limits[:-1]

    return {'par_hard_limits': par_hard_limits,  # no fitting outside
            'par_soft_limits': par_soft_limits,  # no simulations outside
            'default_pars': default_pars,
            'par_names': par_names,
            'fit_par_names': fit_par_names,
            'fit_pars': np.array([par in fit_par_names for par in par_names])}


def get_minimizer_stuff(run_on_cluster, plot_data_path):

    if run_on_cluster:
        return {'save_plot_data': False,
                'plot_data_path': plot_data_path,
                'verbose': False,
                'brute_Ns': 50,
                'hoppin_T': 10.0,
                'hoppin_stepsize': 0.5,
                'NM_niter': 300,
                'NM_xatol': .01,
                'NM_fatol': 1e-6,
                'NM_maxfev': 1000}

    else:
        return {'save_plot_data': False,
                'plot_data_path': plot_data_path,
                'verbose': False,
                'brute_Ns': 10,
                'hoppin_T': 10.0,
                'hoppin_stepsize': 0.5,
                'NM_niter': 10,
                'NM_xatol': .01,
                'NM_fatol': 1e-5,
                'NM_maxfev': 1000}


def get_random_pars(parameters, set_specific_parameters, learning_style):

    # Get random parameters within soft limits, with default pars for those that should not be fit
    gen_pars = np.array([lim[0] + np.random.rand() * (lim[1] - lim[0]) for lim in parameters['par_soft_limits']])
    fixed_par_idx = np.invert(parameters['fit_pars'])
    gen_pars[fixed_par_idx] = parameters['default_pars'][fixed_par_idx]

    # Adjust beta_high and TS_bias if learning style is flat
    beta_high_TS_bias = np.argwhere([par in ['beta_high', 'TS_bias'] for par in parameters['par_names']])
    if learning_style == 'flat':
        gen_pars[beta_high_TS_bias] = [100., 100.]
    elif learning_style == 's-flat':
        gen_pars[beta_high_TS_bias] = [0., 1.]

    # Alternatively, let user set parameters
    if set_specific_parameters:
        for i, par in enumerate(gen_pars):
            change_par = input('Accept {0} of {1}? If not, type a new number.'.
                               format(parameters['par_names'][i], np.round(par, 2)))
            if change_par:
                gen_pars[i] = float(change_par)

    return gen_pars
