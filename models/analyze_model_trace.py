# Model comparison: https://docs.pymc.io/notebooks/model_comparison.html

import pickle
import string
import pandas as pd
import seaborn as sns

import pymc3 as pm
from theano.printing import pydotprint
import matplotlib.pyplot as plt

from shared_modeling_simulation import get_paths
from modeling_helpers import plot_gen_rec


# Which models should be analyzed and compared?
create_pairplot = False
analyze_indiv_models = True
test_group_differences = False
compare_models = True
calculate_waic = True
do_plot_gen_rec = False
param_names = ['alpha', 'beta', 'forget', 'alpha_high', 'beta_high', 'forget_high']

file_names = [
    # 'Bayes_3groups/betswirew_2018_10_10_12_1_humans_n_samples100',
    # 'Bayes_3groups/betswirew_2018_10_9_16_42_humans_n_samples5000',
    # 'Bayes_3groups/betswirew_2018_10_9_16_50_humans_n_samples5000'
    'Aliens/f_abf_2018_10_11_11_31_humans_n_samples1000',
    'Aliens/fs_abf_2018_10_10_17_52_humans_n_samples1000',
    'Aliens/max_abf_2018_10_10_18_9_humans_n_samples1000',
    'Aliens/soft_abf_2018_10_11_11_36_humans_n_samples100'
    ]
model_names = ['f', 'fs', 'max', 'soft']  # ['betswirew', 'swirew', 'betswirew']  # string.ascii_lowercase  # ['abn', 'abcncn']

# Load fitted parameters
paths = get_paths(run_on_cluster=False)
parameter_dir = paths['fitting results']
save_dir = paths['fitting results']
print("Working on {0}.\n".format(file_names))

model_dict = {}
for file_name, model_name in zip(file_names, model_names):

    print('\n\tMODEL {0}'.format(model_name))
    with open(parameter_dir + file_name + '.pickle', 'rb') as handle:
        data = pickle.load(handle)
        trace = data['trace']
        model = data['model']
        model_summary = data['summary']
        model.name = model_name

    if do_plot_gen_rec:
        # map_gen_rec = pd.read_csv(save_dir + file_name + '_map_gen_rec.csv', index_col=0, header=0)
        # plot_gen_rec(param_names=param_names, gen_rec=map_gen_rec,
        #              save_name=save_dir + file_name + '_map_gen_rec_plot.png')

        mcmc_gen_rec = pd.read_csv(save_dir + file_name + '_mcmc_gen_rec.csv', index_col=0, header=None)
        plot_gen_rec(param_names=param_names, gen_rec=mcmc_gen_rec,
                     save_name=save_dir + file_name + '_mcmc_gen_rec_plot.png')

    # Add model to model_dict
    model_dict.update({model: trace})

    if analyze_indiv_models:

        # Compare true parameters and estimates
        # for param_name in ['alpha', 'beta', 'forget', 'alpha_high']:
        #     traces = trace[param_name].reshape(trace['alpha'].shape).T
        #     true_forgets = map_gen_rec.loc[param_name]
        #     colors = plt.cm.PRGn(np.linspace(0, 1, len(true_forgets)))
        #     plt.figure()
        #     for forget_trace, true_forget, color in zip(traces, true_forgets, colors):
        #         sns.kdeplot(forget_trace, color=color)
        #         plt.axvline(true_forget, color=color)
        #         plt.xlabel(param_name)
        #     plt.savefig(save_dir + file_name + param_name + 'MCMC_gen_rec.png')

        par_names = [str(RV_name) for RV_name in model.free_RVs]

        # Graph (does not seem to exist any more in PyMC3)
        # pydotprint(model.logpt)

        # # Plot cumulative means of all parameters
        # for par_name in par_names:
        #     param_trace = trace[par_name]
        #     mparam_trace = [np.mean(param_trace[:i]) for i in np.arange(1, len(param_trace))]
        #     plt.close('all')
        #     plt.figure()
        #     plt.plot(mparam_trace, lw=2.5)
        #     plt.xlabel('Iteration')
        #     plt.ylabel('MCMC mean of {0}'.format(par_name))
        #     plt.title(model_name)
        #     plt.savefig(save_dir + file_name + '_' + par_name + '_cumsumplot.png')

        # display the total number and percentage of divergent
        divergent = trace['diverging']
        print('Number of Divergent %d' % divergent.nonzero()[0].size)

        # Rhat should be close to one; number of effective samples > 200
        print('Saving summary of {0} model.'.format(model_name))
        pd.DataFrame(model_summary).to_csv(save_dir + file_name + '_summary.csv')

        # print(model.basic_RVs)
        # pm.pairplot(trace, sub_varnames=['alpha_sd', 'beta_sd'], divergences=True, color='C3',
        #             kwargs_divergence={'color': 'C2'})
        # plt.savefig(save_dir + file_name + '_diagnostic_pairplot')

        # Plot traces and parameter estimates
        pm.traceplot(trace)
        plt.savefig(save_dir + file_name + '_traceplot.png')
        # pm.forestplot(trace)
        # plt.savefig(save_dir + file_name + '_forestplot.png')
        print("Saved traces for {0} model to {1}{2}.".format(model_name, save_dir, file_name))

        # Get model WAICs
        if calculate_waic:
            waic = pm.waic(trace, model)
            print("WAIC of {0} model: {1}".format(model_name, waic.WAIC))

    # Plot divergent samples to identify problematic neighborhoods in parameter space
    if create_pairplot:
        pm.pairplot(trace,
                    # sub_varnames=[str(par_name) for par_name in par_name_pair],
                    divergences=True,
                    color='C3', kwargs_divergence={'color': 'C2'})
        plt.savefig(save_dir + file_name + '_pairplot.png')

    if test_group_differences:
        diffs = [
            'alpha_a_diff01', 'alpha_a_diff02', 'alpha_a_diff12',
            'alpha_b_diff01', 'alpha_b_diff02', 'alpha_b_diff12',
            'nalpha_a_diff01', 'nalpha_a_diff02', 'nalpha_a_diff12',
            'nalpha_b_diff01', 'nalpha_b_diff02', 'nalpha_b_diff12',
            'calpha_sc_a_diff01', 'calpha_sc_a_diff02', 'calpha_sc_a_diff12',
            'calpha_sc_b_diff01', 'calpha_sc_b_diff02', 'calpha_sc_b_diff12',
            'cnalpha_sc_a_diff01', 'cnalpha_sc_a_diff02', 'cnalpha_sc_a_diff12',
            'cnalpha_sc_b_diff01', 'cnalpha_sc_b_diff02', 'cnalpha_sc_b_diff12'
        ]
        pd.DataFrame(pm.summary(trace, diffs)).to_csv(save_dir + file_name + '_diff_summary.csv')
        diff_plot = pm.plot_posterior(trace, varnames=diffs,
                                      ref_val=0,
                                      color='#87ceeb')
        plt.savefig(save_dir + file_name + '_diffplot.png')

# Compare WAIC scores
if compare_models:
    model_comparison_summary = pm.compare(model_dict)
    pd.DataFrame(model_comparison_summary).to_csv(save_dir + file_name + '_model_comparison_summary.csv')

    pm.compareplot(model_comparison_summary)
    plt.savefig(save_dir + file_name + 'compareplot_WAIC' + '_'.join(model_names) + '.png')
    print("Compared WAICs of {0} models; saved figure to {1}...\n".format(model_names, save_dir))
