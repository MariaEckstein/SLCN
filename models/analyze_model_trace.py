# Model comparison: https://docs.pymc.io/notebooks/model_comparison.html

import pickle
import pandas as pd

import pymc3 as pm
from theano.printing import pydotprint
import matplotlib.pyplot as plt

from shared_modeling_simulation import *


# Which models should be analyzed and compared?
create_pairplot = False

file_names = ['Bayes_groups/Bayes_switch_reward_group_test_2018_7_21_12_5_humans_n_samples1000Bayes',
              'Bayes_beta_priors/Bayes_switch_reward_2018_7_21_11_16_humans_n_samples5000Bayes']
model_names = ['sw_re_gr', 'sw_re']
# file_names = ['RL_beta_priors/RL_alpha_beta_2018_7_20_17_29_humans_n_samples4000RL',
#               'RL_beta_priors/RL_alpha_beta_eps_2018_7_20_17_28_humans_n_samples4000RL',
#               'RL_beta_priors/RL_alpha_beta_nalpha_2018_7_20_17_34_humans_n_samples4000RL',
#               'RL_beta_priors/RL_alpha_beta_calpha_2018_7_20_17_32_humans_n_samples4000RL',
              # 'RL_indep_scalers/RL_alpha_beta_2018_7_20_11_2_humans_n_samples2000RL',
              # ]
# model_names = ['al_bet', 'al_bet_eps', 'al_bet_nal', 'al_bet_cal']

# Load fitted parameters
paths = get_paths(run_on_cluster=False)
parameter_dir = paths['fitting results']
save_dir = paths['fitting results']
print("Loading models {0}.\n".format(file_names))

model_dict = {}
for file_name, model_name in zip(file_names, model_names):
    plt.close('all')
    print('\n\tMODEL {0}'.format(model_name))
    with open(parameter_dir + file_name + '.pickle', 'rb') as handle:
        data = pickle.load(handle)
        trace = data['trace']
        model = data['model']
        model_summary = data['summary']
        model.name = model_name

    par_names = [str(RV_name) for RV_name in model.free_RVs]

    # Graph (does not seem to exist any more in PyMC3)
    # pydotprint(model.logpt)

    # Plot cumulative means of all parameters
    for par_name in par_names:
        param_trace = trace[par_name]
        mparam_trace = [np.mean(param_trace[:i]) for i in np.arange(1, len(param_trace))]
        plt.figure()
        plt.plot(mparam_trace, lw=2.5)
        plt.xlabel('Iteration')
        plt.ylabel('MCMC mean of {0}'.format(par_name))
        plt.title(model_name)
        plt.savefig(save_dir + file_name + '_' + par_name + '_cumsumplot.png')

    # display the total number and percentage of divergent
    divergent = trace['diverging']
    print('Number of Divergent %d' % divergent.nonzero()[0].size)

    # Rhat should be close to one; number of effective samples > 200
    print('Saving summary of {0} model.'.format(model_name))
    pd.DataFrame(model_summary).to_csv(save_dir + file_name + 'model_summary.csv')

    # Plot traces and parameter estimates
    pm.traceplot(trace)
    plt.savefig(save_dir + file_name + '_traceplot.png')
    pm.forestplot(trace)
    plt.savefig(save_dir + file_name + '_forestplot.png')
    print("Saved traces for {0} model to {1}{2}.".format(model_name, save_dir, file_name))

    # Get model WAICs
    waic = pm.waic(trace, model)
    print("WAIC of {0} model: {1}".format(model_name, waic.WAIC))

    # Add model to model_dict
    model_dict.update({model: trace})

    # Plot divergent samples to identify problematic neighborhoods in parameter space
    if create_pairplot:
        pm.pairplot(trace,
                    # sub_varnames=[str(par_name) for par_name in par_name_pair],
                    divergences=True,
                    color='C3', kwargs_divergence={'color': 'C2'})
        plt.savefig(save_dir + file_name + '_pairplot.png')

# Compare WAIC scores
model_comparison_summary = pm.compare(model_dict)
pd.DataFrame(model_comparison_summary).to_csv(save_dir + file_name + '_model_comparison_summary.csv')

pm.compareplot(model_comparison_summary)
plt.savefig(save_dir + file_name + 'compareplot_WAIC' + '_'.join(model_names) + '.png')
print("Compared WAICs of {0} models; saved figure to {1}...\n".format(model_names, save_dir))
