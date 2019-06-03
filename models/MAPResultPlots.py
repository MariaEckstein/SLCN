import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle

from shared_modeling_simulation import get_paths


save_dir = 'C:/Users/maria/MEGAsync/SLCN/PShumanData/fitting/map_indiv/'
n_subj = 233

# Plot BICs of all models
nll_bic = pd.read_csv(save_dir + 'nll_bics.csv')
nll_bic = nll_bic.sort_values(by=['bic'])
plt.figure()
plt.bar(nll_bic['model_name'], nll_bic['bic'])
plt.xticks(rotation='vertical')
plt.ylabel('BIC')
plt.savefig("{0}plot_bics.png".format(save_dir))

# Plot NLLs of all models
nll_bic = nll_bic.sort_values(by=['nll'])
plt.figure()
plt.bar(nll_bic['model_name'], nll_bic['nll'])
plt.xticks(rotation='vertical')
plt.ylabel('NLL')
plt.savefig("{0}plot_nlls.png".format(save_dir))

# Bring parameter estimates in the right shape
file_names = [f for f in os.listdir(save_dir) if '.pickle' in f]
ages = pd.read_csv(get_paths(False)['ages'], index_col=0)
for file_name in file_names:
    with open(save_dir + file_name, 'rb') as handle:

        # Load pickled model fitting results
        data = pickle.load(handle)
        map = data['map']

        # Fitted parameters are stored in dictionaries, with parameter names as keys
        param_names = [key for key in map.keys() if '__' not in key]
        fitted_params = []

        # Create csv files with the fitted parameters for easier access
        for param_name in param_names:
            new_row = map[param_name] * np.ones(n_subj)
            fitted_params.append(new_row.reshape(n_subj))
        fitted_params = np.array(fitted_params).T
        fitted_params = pd.DataFrame(fitted_params, columns=param_names)
        fitted_params['sID'] = ages['sID']  # index of ages == PyMC file order == fitted_params order

        # Save the csvs to save_dir
        print("Saving csv of fitted parameters for model {0} to {1}.".format(file_name, save_dir))
        fitted_params.to_csv(save_dir + 'params_' + file_name[:-7] + '.csv', index=False)
