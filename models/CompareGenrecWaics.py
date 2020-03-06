import itertools
import numpy as np
import os
import plotnine as gg
gg.theme_set(gg.theme_bw)
import pandas as pd
import pickle
import pymc3 as pm


# COMPARE WAICS
################

# # Directories
# # pickle_path = 'C:/Users/maria/MEGAsync/SLCN/PShumanData/fitting/new_ML_models/MCMC/clustermodels/genrec/new/'
# pickle_path = 'C:/Users/maria/MEGAsync/SLCN/PShumanData/fitting/new_ML_models/MCMC/clustermodels/'
#
# # Read in pickled models
# file_names = [f for f in os.listdir(pickle_path) if '.pickle' in f]
# # file_names = [f for f in os.listdir(pickle_path) if ('.pickle' in f) and ('RL' in f) and ('y' not in f)]
# # file_names = sorted(file_names, key=len)  # sort by number of parameters
# file_names = [file_names[5], file_names[15]]
#
# waics = {}
# models = {}
# traces = {}
# model_names = []
#
# for file_name in file_names:
#     model_name = [part for part in file_name.split('_') if ('RL' in part) or ('B' in part) or ('WSLS' in part)][0]
#     model_names.append(model_name)
#     print(model_name)
#
#     with open(os.path.join(pickle_path, file_name), 'rb') as handle:
#         data = pickle.load(handle)
#         waics.update({model_name: data['WAIC']})
#         models.update({model_name: data['model']})
#         traces.update({model_name: data['trace']})
#
# # Compare WAICs
# print("Comparing models.")
# comps = [pm.compare({models[model_names[i]]: traces[model_names[i]],
#                      models[model_names[i+1]]: traces[model_names[i+1]]})
#          for i in range(len(file_names)-1)]
#
# # comps = comps.append(pm.compare({models[model_names[3]]: traces[model_names[3]],
# #                      models[model_names[5]]: traces[model_names[5]]}))
#
# # comp_BF = pm.compare({models[model_names[2]]: traces[model_names[2]],
# #                       models[model_names[3]]: traces[model_names[3]]})
#
# stop = 4


# CALCULATE AGE DIFFERENCSE IN MODEL PARAMETERS
################################################

# Directories
pickle_path = 'C:/Users/maria/MEGAsync/SLCN/PShumanData/fitting/new_ML_models/MCMC/clustermodels/'
ages_path = 'C:/Users/maria/MEGAsync/SLCNdata/ages.csv'

ages = pd.read_csv(ages_path)

# Read in pickled models
file_names = [f for f in os.listdir(pickle_path) if '.pickle' in f]
file_names = [file_names[5], file_names[15]]

traces = {}
model_names = []

for file_name in file_names:
    model_name = [part for part in file_name.split('_') if ('RL' in part) or ('B' in part) or ('WSLS' in part)][0]
    model_names.append(model_name)
    print(model_name)

    with open(os.path.join(pickle_path, file_name), 'rb') as handle:
        data = pickle.load(handle)
        traces.update({model_name: data['trace']})


def add_quantile_groups(data, col_name):

    qcol_name = col_name + '_quant'
    data[qcol_name] = np.nan

    # Set adult quantile to 2
    data.loc[data.sID > 300, qcol_name] = 3  # Adults
    data.loc[data.sID > 400, qcol_name] = 2  # UCB

    # Determine quantiles, separately for both genders
    for gender in np.unique(data['gender']):

        # Get 4 quantiles
        cut_off_values = np.nanquantile(
            data.loc[(data.age < 20) & (data.gender == gender), col_name],
            [0, 1/4, 2/4, 3/4])
        for cut_off_value, quantile in zip(cut_off_values, np.round([1/4, 2/4, 3/4, 1], 2)):
            data.loc[
                (data.age < 20) & (data[col_name] >= cut_off_value) & (data.gender == gender),
                qcol_name] = quantile

# Calculate differences
def get_param_dat(traces, model_name, param_name):

    param_dat = traces[model_name][param_name]
    param_dat = pd.DataFrame(param_dat.T)
    param_dat['sID'] = data['sIDs']
    param_dat = param_dat.merge(ages)
    add_quantile_groups(param_dat, 'age')

    return param_dat

def get_p_values(param_dat):

    mean_kids = np.mean(param_dat[param_dat.age_quant == 0.25])
    mean_teens = np.mean(param_dat[param_dat.age_quant == 0.75])
    mean_adults = np.mean(param_dat[param_dat.age_quant == 3])

    teens_min_kids = mean_teens - mean_kids
    adults_min_teens = mean_adults - mean_teens

    return pd.DataFrame(
        {'p_kids_teens': [np.mean(teens_min_kids[:10000] > 0)],
         'p_adults_teens': [np.mean(adults_min_teens[:10000] < 0)]})


p_BF = get_param_dat(traces, 'Bbspr', 'persev')
beta_BF = get_param_dat(traces, 'Bbspr', 'beta')
p_reward = get_param_dat(traces, 'Bbspr', 'p_reward')
p_switch = get_param_dat(traces, 'Bbspr', 'p_switch')

p_RL = get_param_dat(traces, 'RLabnp2', 'persev')
beta_RL = get_param_dat(traces, 'RLabnp2', 'beta')
nalpha = get_param_dat(traces, 'RLabnp2', 'nalpha')
alpha = get_param_dat(traces, 'RLabnp2', 'alpha')

get_p_values(p_BF)
get_p_values(beta_BF)
get_p_values(p_reward)
get_p_values(p_switch)

get_p_values(p_RL)
get_p_values(beta_RL)
get_p_values(nalpha)
get_p_values(p_RL)

stop = 4
