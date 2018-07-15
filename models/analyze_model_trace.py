# Model comparison: https://docs.pymc.io/notebooks/model_comparison.html

import pickle

import pymc3 as pm
import matplotlib.pyplot as plt
from collections import defaultdict

from shared_modeling_simulation import *


# Which models should be analyzed and compared?
file_names = ['test_sd_fitting_2018_7_13_9_49_humans_n_samples5000RL',
              'test_sd_fitting_2018_7_13_9_49_humans_n_samples5000Bayes']
model_names = ['RL', 'Bayes']

# Load fitted parameters
paths = get_paths(run_on_cluster=False)
parameter_dir = paths['fitting results']
save_dir = paths['fitting results']
print("Loading models... files: {0}\n".format(file_names))


# A small wrapper function for displaying the MCMC sampler diagnostics as above
def report_trace(trace, par_names):

    # plot the trace of parameters
    pm.traceplot(trace, varnames=par_names, transform=np.log)
    plt.show()

    # plot the estimate for the mean of parameter cumulating mean
    param_trace = trace[par_names[0]]
    mparam_trace = [np.mean(param_trace[:i]) for i in np.arange(1, len(param_trace))]
    plt.plot(mparam_trace, lw=2.5)
    plt.xlabel('Iteration')
    plt.ylabel('MCMC mean of {0}'.format(par_names[0]))
    plt.title('MCMC estimation')
    plt.show()

    # display the total number and percentage of divergent
    divergent = trace['diverging']
    print('Number of Divergent %d' % divergent.nonzero()[0].size)
    divperc = divergent.nonzero()[0].size / len(trace) * 100
    print('Percentage of Divergent %.1f' % divperc)

    # scatter plot for the identifcation of the problematic neighborhoods in parameter space
    pm.pairplot(trace,
                sub_varnames=par_names,
                divergences=True,
                color='C3', kwargs_divergence={'color': 'C2'})
    plt.show()


model_dict = {}
for file_name, model_name in zip(file_names, model_names):
    with open(parameter_dir + file_name + '.pickle', 'rb') as handle:
        data = pickle.load(handle)
        trace = data['trace']
        model = data['model']
        model.name = model_name

    # # Graph
    # pm.graph

    # Check for geometric ergodicity
    print(pm.summary(trace).round(2))  # Rhat should be close to one; number of effective samples > 200

    # Plot divergent samples
    print(model.basic_RVs)
    report_trace(trace, ['eps_mu_interval__', 'eps_sd_log__'])

    # Plot traces and parameter estimates
    print("Plotting traces for {0} model...\n".format(model_name))
    pm.traceplot(trace)
    plt.savefig(save_dir + 'traceplot' + model_name + '.png')
    pm.forestplot(trace)
    plt.savefig(save_dir + 'forestplot' + model_name + '.png')

    # Get model WAICs
    waic = pm.waic(trace, model)
    print(waic.WAIC)

    # Add model to model_dict
    model_dict.update({model: trace})

# Compare WAIC scores
print("Comparing models using WAIC...\n")
pm.compareplot(pm.compare(model_dict))
plt.savefig(save_dir + 'compareplot_WAIC' + '_'.join(model_names) + '.png')
