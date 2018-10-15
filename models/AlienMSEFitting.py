from itertools import combinations
import datetime
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, brute, basinhopping

from shared_aliens import alien_initial_Q, update_Qs_sim
from AlienTask import Task
from basinhopping_specifics import MyBounds, MyTakeStep

n_subj = 31  # 31
n_sim_per_subj = 50  # 10
n_TS, n_seasons, n_aliens, n_actions = 3, 3, 4, 3  # 3, 3, 4, 3
model_name = 'fs'
param_scalers = pd.DataFrame.from_dict(
    {'alpha': [1], 'beta': [10], 'forget': [1]})  # , 'alpha_high': [0.3]})  # , 'beta_high': [10], 'forget_high': [0.1]})
param_names = np.array(param_scalers.columns.values)
plot_dir = 'C:/Users/maria/MEGAsync/SLCN/PShumanData/fitting'

# Split subjects into half
half_of_subj = np.arange(0, n_subj, 2)  # np.random.choice(range(n_subj), size=int(np.ceil(n_subj / 2)), replace=False)
other_half = [i for i in range(n_subj) if i not in half_of_subj]
half_of_subj = half_of_subj[:len(other_half)]
n_sim = len(half_of_subj) * n_sim_per_subj  # n_subj * n_sim_per_subj

# Minimizer options
n_iter = 10
nelder_mead_options = {
    'maxfev': 50,
    'xatol': 0.01,
    'fatol': 0.01
}

# Make save_id
now = datetime.datetime.now()
save_id = '{0}_{1}_{2}_{3}'.format(model_name, param_names, param_scalers.values, '_'.join([str(i) for i in [now.year, now.month, now.day, now.hour, now.minute]]))
print("running {}".format(save_id))

# Parameter shapes
beta_shape = (n_sim, 1)  # Q_low_sub.shape -> [n_sim, n_actions]
forget_shape = (n_sim, 1, 1, 1)  # Q_low[0].shape -> [n_sim, n_TS, n_aliens, n_actions]
beta_high_shape = (n_sim, 1)  # Q_high_sub.shape -> [n_sim, n_TS]
forget_high_shape = (n_sim, 1, 1)  # -> [n_sim, n_seasons, n_TS]


def calculate_mse(parameters, param_scalers, subset_of_subj, make_plot=False):

    # Rescale parameters
    parameters = param_scalers * parameters

    if 'alpha' in parameters:
        alpha = parameters['alpha'].values * np.ones(n_sim)
    else:
        alpha = 0.2 * np.ones(n_sim)

    if 'beta' in parameters:
        beta = parameters['beta'].values * np.ones(beta_shape)
    else:
        beta = 2 * np.ones(beta_shape)

    if 'forget' in parameters:
        forget = parameters['forget'].values * np.ones(forget_shape)
    else:
        forget = 0.01 * np.ones(forget_shape)

    if 'alpha_high' in parameters:
        alpha_high = parameters['alpha_high'].values * np.ones(n_sim)
    else:
        alpha_high = alpha.copy()

    if 'beta_high' in parameters:
        beta_high = parameters['beta_high'].values * np.ones(beta_high_shape)
    else:
        beta_high = beta.flatten().reshape(beta_high_shape)

    if 'forget_high' in parameters:
        forget_high = parameters['forget_high'].values * np.ones(forget_high_shape)
    else:
        forget_high = forget.flatten().reshape(forget_high_shape)

    # Initialize stuff
    task = Task(n_subj)
    n_trials, hum_corrects = task.get_trial_sequence("C:/Users/maria/MEGAsync/Berkeley/TaskSets/Data/version3.1/",
                                                     n_subj, n_sim_per_subj, subset_of_subj, phases=("1InitialLearning", ""))

    sim_corrects = np.zeros([n_trials, n_sim])

    Q_low = alien_initial_Q * np.ones([n_sim, n_TS, n_aliens, n_actions])
    Q_high = alien_initial_Q * np.ones([n_sim, n_seasons, n_TS])

    for trial in range(n_trials):

        # Observe stimuli
        season, alien = task.present_stimulus(trial)

        # Select action & update Q-values
        [Q_low, Q_high, TS, action, correct, reward, p_low] =\
            update_Qs_sim(season, alien,
                          Q_low, Q_high,
                          beta, beta_high, alpha, alpha_high, forget, forget_high,
                          n_sim, n_actions, n_TS, task, verbose=False)

        # Get data for learning curves
        sim_corrects[trial] = correct

    # Calculate group learning curves and MSE
    sim_learning_curve = np.mean(sim_corrects, axis=1)
    hum_learning_curve = np.mean(hum_corrects, axis=1)
    MSE = np.mean((hum_learning_curve - sim_learning_curve) ** 2)

    # To check minimizer convergence
    print("fitting {0}; current MSE {1}".format(parameters.round(3), MSE.round(3)))

    # Plot learning curves
    if make_plot:
        plt.subplot(3, 1, 1)
        plt.errorbar(x=np.arange(n_trials),
                     y=sim_learning_curve,
                     yerr=np.std(sim_corrects, axis=1) / np.sqrt(sim_corrects.shape[1]))
        plt.title('Simulation ({})'.format(parameters.round(2)))
        plt.ylim((0, 1))
        plt.subplot(3, 1, 2)
        plt.errorbar(x=np.arange(n_trials),
                     y=hum_learning_curve,
                     yerr=np.std(hum_corrects, axis=1) / np.sqrt(hum_corrects.shape[1]))
        plt.title('Humans')
        plt.ylim((0, 1))
        plt.subplot(3, 1, 3)
        plt.plot(np.arange(n_trials),
                 hum_learning_curve - sim_learning_curve, '-')
        plt.axhline(0)
        plt.title('Difference (MSE {})'.format(MSE.round(3)))
        plt.ylim((-0.5, 0.5))
        plt.tight_layout()

    return MSE


# Minimize function using grid search
print("Starting brute with {} iterations!".format(n_iter))
# Needs >= 3 param_names; speed up for debugging: set n_subj=2 and n_sim_per_subj=2
brute_results = brute(func=calculate_mse,
                      ranges=([(1e-3, 1-1e-3) for param in param_names]),
                      args=([param_scalers, half_of_subj]),
                      Ns=n_iter,
                      full_output=True,
                      finish=None,
                      disp=True)

print('Saving brute_results to {0}/{1}.\n'.format(plot_dir, save_id))
with open(plot_dir + '/' + save_id + '.pickle', 'wb') as handle:
    pickle.dump(brute_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Plot learning curve at minimum
plt.figure()
calculate_mse(brute_results[0], param_scalers, half_of_subj, make_plot=True)
plt.savefig(plot_dir + '/in_sample_{}.png'.format(save_id))
plt.figure()
calculate_mse(brute_results[0], param_scalers, other_half, make_plot=True)
plt.savefig(plot_dir + '/out_of_sample_{}.png'.format(save_id))

# Plot heatmap for the whole space
combos = list(combinations(param_names, 2))
n = len(param_names) - 1
tril_pos = np.tril((np.arange(n ** 2) + 1).reshape(n, -1)).T.ravel()
positions = tril_pos[tril_pos != 0]

plt.figure()
for (xname, yname), pos in zip(combos, positions):

    # Specify subplot
    ax = plt.subplot(n, n, pos)

    # Find index for the two parameters
    xi = int(np.argwhere(param_names == xname))  # param_names.index(xname)
    yi = int(np.argwhere(param_names == yname))  # param_names.index(yname)

    # get the meshgrids for x and y
    X = brute_results[2][xi]
    Y = brute_results[2][yi]

    # Find other axes to collapse
    axes = tuple([ii for ii in range(brute_results[3].ndim) if ii not in (xi, yi)])

    # Collapse to minimum Jout
    min_jout = np.amin(brute_results[3], axis=axes)
    min_xgrid = np.amin(X, axis=axes)
    min_ygrid = np.amin(Y, axis=axes)

    # Create heatmap
    ax.pcolormesh(min_xgrid, min_ygrid, min_jout)

    # Add labels to edge only
    if pos >= n ** 2 - n:
        plt.xlabel(xname)
    if pos % n == 1:
        plt.ylabel(yname)

plt.tight_layout()
plt.savefig(plot_dir + '/heatmap_{}.png'.format(save_id))

# print("Starting basinhopping!")
# # Can be sped up by just fitting one parameter; needs a good number of n_subj and n_sim_per_subj to work properly
# bounds = MyBounds(xmax=np.ones(len(param_names)), xmin=np.zeros(len(param_names)))
# takestep = MyTakeStep(stepsize=0.5)
# hoppin_results = basinhopping(func=calculate_mse,
#                               x0=.5 * np.ones(len(param_names)),
#                               niter=n_iter,
#                               # T: The “temperature” parameter for the accept or reject criterion. Higher “temperatures” mean that larger jumps in function value will be accepted. For best results T should be comparable to the separation (in function value) between local minima.
#                               T=0.01,
#                               minimizer_kwargs={'method': 'Nelder-Mead',
#                                                 'args': param_scalers,
#                                                 'options': nelder_mead_options
#                                                 },
#                               take_step=takestep,
#                               accept_test=bounds,
#                               disp=True)
# hoppin_fit_par, hoppin_MSE = [hoppin_results.x, hoppin_results.fun]
# print("Found minimum {0} with MSE {1} through basinhopping!".format(hoppin_fit_par.round(2), hoppin_MSE.round(4)))
# calculate_mse(hoppin_fit_par, param_scalers, make_plot=True)

# print("Starting minimize!")
# # Needs many n_subj and n_sim_per_subj to be able to find a minimum; n_iter needs to be >= 2
# all_minimize_res = np.full((n_iter, len(param_names)+1), np.nan)
# for i in range(n_iter):
#     print("minimize iteration {}".format(i))
#     random_start_values = np.random.rand(len(param_names))
#     minimize_res = minimize(fun=calculate_mse,
#                             x0=random_start_values,
#                             args=param_scalers,
#                             tol=0.01,
#                             method='Nelder-Mead',
#                             options=nelder_mead_options)
#     all_minimize_res[i] = np.concatenate([[calculate_mse(minimize_res.x, param_names)], minimize_res.x])
#
# minimize_results = all_minimize_res[all_minimize_res[:, 0] == np.min(all_minimize_res[:, 0])].flatten()
# calculate_mse(minimize_results[1:], param_scalers, make_plot=True)
# print("Found minimum {0} with MSE {1} through minimize!".
#       format(minimize_results[1:].round(3), minimize_results[0].round(4)))
