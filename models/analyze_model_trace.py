# Model comparison: https://docs.pymc.io/notebooks/model_comparison.html

import pickle

import pymc3 as pm

from shared_modeling_simulation import Shared


# Initialize shared
shared = Shared(None)

# Load fitted parameters
print("Loading models")
parameter_dir = shared.get_paths()['fitting results']
with open(parameter_dir + 'trace_2018_7_2_nsubj155Hier_mu.pickle', 'rb') as handle:
    trace = pickle.load(handle)
with open(parameter_dir + 'model_2018_7_2_nsubj155Hier_mu.pickle', 'rb') as handle:
    model = pickle.load(handle)

# Plots
pm.traceplot(trace)
pm.forestplot(trace)

# Get model model WAICs
waic = pm.waic(trace, model)
waic.WAIC

# # Compare WAIC scores
# df_comp_WAIC = pm.compare({model_simple_flat: trace_simple_flat,
#                            model_counter_flat: trace_counter_flat})
#
# pm.compareplot(df_comp_WAIC)
#
# # Compare leave-one-out cross validation
# df_comp_LOO = pm.compare({model_simple_flat: trace_simple_flat,
#                           model_counter_flat: trace_counter_flat}, ic='LOO')
#
# pm.compareplot(df_comp_LOO)

stop = True
