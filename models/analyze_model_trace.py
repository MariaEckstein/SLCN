# Model comparison: https://docs.pymc.io/notebooks/model_comparison.html

import pickle

import pymc3 as pm
import matplotlib.pyplot as plt

from shared_modeling_simulation import Shared


# Which models should be analyzed and compared?
model1_name = '2018_7_10_16_11_humans_RL_hierarchical_n_samples10000'
model2_name = '2018_7_7_23_17_humans_Bayes_hierarchical_n_samples10000'

# Initialize shared
shared = Shared(run_on_cluster=False)

# Load fitted parameters
parameter_dir = shared.get_paths()['fitting results']
print("Loading models... (names: {0} and {1})\n".format(model1_name, model2_name))

with open(parameter_dir + model1_name + '.pickle', 'rb') as handle:
    RL_hier_data = pickle.load(handle)
    RL_hier_trace = RL_hier_data['trace']
    RL_hier_model = RL_hier_data['model']
    RL_hier_model.name = 'RL_hier'

with open(parameter_dir + model2_name + '.pickle', 'rb') as handle:
    RL_flat_data = pickle.load(handle)
    RL_flat_trace = RL_flat_data['trace']
    RL_flat_model = RL_flat_data['model']
    RL_flat_model.name = 'RL_flat'

# Plots
print("Plotting traces for the first model...\n")
pm.traceplot(RL_hier_trace)
plt.show()
pm.forestplot(RL_hier_trace)
plt.show()

# Get model WAICs
waic = pm.waic(RL_hier_trace, RL_hier_model)
print(waic.WAIC)

# Compare WAIC scores
print("Comparing models using WAIC...\n")
df_comp_WAIC = pm.compare({RL_flat_model: RL_flat_trace,
                           RL_hier_model: RL_hier_trace})

pm.compareplot(df_comp_WAIC)
plt.show()

# Compare leave-one-out cross validation
print("Comparing models using LOO...\n")
df_comp_LOO = pm.compare({RL_flat_model: RL_flat_trace,
                          RL_hier_model: RL_hier_trace}, ic='LOO')

pm.compareplot(df_comp_LOO)
plt.show()
