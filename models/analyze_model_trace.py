# Model comparison: https://docs.pymc.io/notebooks/model_comparison.html

import pickle
import pandas as pd

import pymc3 as pm
from theano.printing import pydotprint
import matplotlib.pyplot as plt

from shared_modeling_simulation import *


# Which models should be analyzed and compared?
create_pairplot = False
analyze_indiv_models = True
test_group_differences = False
compare_models = False

# file_names = ['RL_3groups/TestGroupDifferences/albenalcal_2018_7_27_21_19_humans_n_samples5000RL',
              # 'Bayes_groups/rew_swi_eps20_2018_7_24_19_16_humans_n_samples5000Bayes'
              # ]
# model_names = ['albenalcal', 'rew_swi_eps']
file_names = [
    'AliensFlat/flat_abf_2018_8_9_16_1_humans_n_samples2000aliens',
    'AliensFlat/flat_ab_2018_8_9_16_19_humans_n_samples200aliens',
    # 'AliensFlat/flat_2018_8_8_17_21_humans_n_samples200aliens',
    # 'RL_3groups/TestGroupDifferences/alpha_beta20_2018_7_24_19_4_humans_n_samples5000RL',
    # 'RL_3groups/100s/al_be_cal100_2018_7_25_21_40_humans_n_samples5000RL',
    # 'RL_3groups/100s/al_be_nal_cal100_2018_7_25_21_40_humans_n_samples5000RL',
#     'RL_3groups/100s/al_be_nal100_2018_7_25_21_38_humans_n_samples5000RL',
#     'RL_3groups/100s/alpha_beta100_2018_7_25_21_20_humans_n_samples5000RL',
#     'RL_3groups/10s/alpha_beta_2018_7_24_14_11_humans_n_samples5000RL',
#     'RL_3groups/10s/alpha_beta_calpha_2018_7_24_14_15_humans_n_samples5000RL',
    # 'RL_3groups/10s/alpha_beta_nalpha_calpha_eps_2018_7_24_18_46_humans_n_samples5000RL'
              ]
model_names = ['abf', 's', '...', 'albe20', 'albecal100', 'albenalcal100', 'albenal100', 'albe100', 'albe10', 'albecal10', 'albenalcaleps10']

# Load fitted parameters
paths = get_paths(run_on_cluster=False)
parameter_dir = paths['fitting results']
save_dir = paths['fitting results']
print("Loading models {0}.\n".format(file_names))

model_dict = {}
for file_name, model_name in zip(file_names, model_names):

    print('\n\tMODEL {0}'.format(model_name))
    with open(parameter_dir + file_name + '.pickle', 'rb') as handle:
        data = pickle.load(handle)
        trace = data['trace']
        model = data['model']
        model_summary = data['summary']
        model.name = model_name

    # Add model to model_dict
    model_dict.update({model: trace})

    if analyze_indiv_models:
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
        pd.DataFrame(model_summary).to_csv(save_dir + file_name + 'model_summary.csv')

        #
        # print(model.basic_RVs)
        pm.pairplot(trace, sub_varnames=['alpha_sd_sd', 'beta_sd_sd'], divergences=True, color='C3',
                    kwargs_divergence={'color': 'C2'})
        plt.savefig(save_dir + file_name + '_diagnostic_pairplot')

        # Plot traces and parameter estimates
        pm.traceplot(trace)
        plt.savefig(save_dir + file_name + '_traceplot.png')
        pm.forestplot(trace)
        plt.savefig(save_dir + file_name + '_forestplot.png')
        print("Saved traces for {0} model to {1}{2}.".format(model_name, save_dir, file_name))

        # Get model WAICs
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
