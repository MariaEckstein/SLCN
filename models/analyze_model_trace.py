# Model comparison: https://docs.pymc.io/notebooks/model_comparison.html

import pickle
from itertools import combinations

import pymc3 as pm
from theano.printing import pydotprint
import matplotlib.pyplot as plt

from shared_modeling_simulation import *


def report_trace(trace, par_names):

    # A small wrapper function for displaying the MCMC sampler diagnostics as above

    # plot the estimate for the mean of parameter cumulating mean
    for par_name in par_names:
        param_trace = trace[par_name]
        mparam_trace = [np.mean(param_trace[:i]) for i in np.arange(1, len(param_trace))]
        plt.plot(mparam_trace, lw=2.5)
        plt.xlabel('Iteration')
        plt.ylabel('MCMC mean of {0}'.format(par_names[0]))
        plt.title('MCMC estimation of {0}'.format(par_name))
        plt.show()

    # display the total number and percentage of divergent
    divergent = trace['diverging']
    print('Number of Divergent %d' % divergent.nonzero()[0].size)
    divperc = divergent.nonzero()[0].size / len(trace) * 100
    print('Percentage of Divergent %.1f' % divperc)

    # scatter plot for the identifcation of the problematic neighborhoods in parameter space
    pm.pairplot(trace,
                sub_varnames=[str(par_name) for par_name in par_names[0:2]],
                divergences=True,
                color='C3', kwargs_divergence={'color': 'C2'})
    plt.show()


# Which models should be analyzed and compared?
file_names = ['d715/RL_nalpha_Bayes_beta_target_accept80_2018_7_15_1_28_humans_n_samples1000Bayes',
              'd715/RL_nalpha_Bayes_beta_target_accept80_2018_7_15_1_28_humans_n_samples1000RL']
model_names = ['Bayes', 'RL']

# Load fitted parameters
paths = get_paths(run_on_cluster=False)
parameter_dir = paths['fitting results']
save_dir = paths['fitting results']
print("Loading models {0} and {1}.\n".format(*file_names))

model_dict = {}
for file_name, model_name in zip(file_names, model_names):
    with open(parameter_dir + file_name + '.pickle', 'rb') as handle:
        data = pickle.load(handle)
        trace = data['trace']
        model = data['model']
        trace_summary = data['summary']
        model.name = model_name

    par_names = [str(RV_name) for RV_name in model.free_RVs]

    # Graph (does not seem to exist any more in PyMC3)
    # pydotprint(model.logpt)

    # plot the estimate for the mean of parameter cumulating mean
    # n_rows_cols = int(np.ceil(np.sqrt(len(par_names))))
    # fig, axes = plt.subplots(figsize=[10, 6], nrows=n_rows_cols, ncols=n_rows_cols)
    # ax = list(axes.ravel())
    # idx = 0
    # for par_name in par_names:
    #     param_trace = trace[par_name]
    #     # print(par_name, param_trace[0])
    #     mparam_trace = [np.mean(param_trace[:i]) for i in np.arange(1, len(param_trace))]
    #     ax[idx].plot(mparam_trace, lw=2.5)
    #     ax[idx].set(xlabel='Iteration',
    #                 ylabel=par_name)
    #     idx += 1
    # plt.show()

    # fig, axes = plt.subplots(nrows=3, ncols=3)
    # ax = list(axes.ravel())
    # idx = 0
    # for par_name in par_names:
    #     param_trace = trace[par_name]
    #     mparam_trace = [np.mean(param_trace[:i]) for i in range(1, len(param_trace))]
    #     ax[idx].plot(param_trace)
    #     ax[idx].plot(mparam_trace)
    #     idx += 1
    # plt.show()

    # n_rows_cols = np.ceil(np.sqrt(len(par_names)))
    # fig, axes = plt.subplots(figsize=[10, 6], nrows=n_rows_cols, ncols=n_rows_cols)
    # ax = list(axes.ravel())
    # idx = 0
    # for par_name in par_names:
    #     param_trace = trace[par_name]
    #     mparam_trace = [np.mean(param_trace[:i]) for i in np.arange(1, len(param_trace))]
    #     ax[idx].plot(mparam_trace, lw=2.5)
    #     # plt.xlabel('Iteration')
    #     # plt.ylabel('MCMC mean of {0}'.format(par_name))
    #     ax[idx].set(title='MCMC estimation of {0} in {1} model'.format(par_name, model_name),
    #                 xlabel='Iteration',
    #                 ylabel='MCMC mean of {0}'.format(par_name))
    #     idx += 1
    # plt.show()

    # Check for geometric ergodicity
    print('Summary of {0} model:\n{1}'.format(model_name, trace_summary.round(2)))  # Rhat should be close to one; number of effective samples > 200

    # Plot divergent samples
    for par_name_pair in combinations(par_names, 2):
        report_trace(trace, par_name_pair)

    # Plot traces and parameter estimates
    pm.traceplot(trace)
    plt.savefig(save_dir + file_name + '_traceplot' + '.png')
    pm.forestplot(trace)
    plt.savefig(save_dir + file_name + '_forestplot' + '.png')
    print("Saved traces for {0} model to {1}{2}.\n".format(model_name, save_dir, file_name))

    # Get model WAICs
    waic = pm.waic(trace, model)
    print("WAIC of {0} model: {1}".format(model_name, waic.WAIC))

    # Add model to model_dict
    model_dict.update({model: trace})

# Compare WAIC scores
pm.compareplot(pm.compare(model_dict))
plt.savefig(save_dir + 'compareplot_WAIC' + '_'.join(model_names) + '.png')
print("Compared WAICs of {0} and {1} model; saved figure to {2}...\n".format(*model_names, save_dir))
