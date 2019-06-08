import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns
sns.set(style='whitegrid')

from shared_modeling_simulation import get_paths


save_dir = 'C:/Users/maria/MEGAsync/SLCN/PShumanData/fitting/map_indiv/'
n_subj = 233

# Read in nll_bic
nll_bic = pd.read_csv(save_dir + 'nll_bics.csv')
nll_bic['color'] = nll_bic['model_name'].astype(str).str[:1]

# Plot BICs of all models
nll_bic = nll_bic.sort_values(by=['bic'])
plt.figure(figsize=(15, 10))
ax = sns.barplot(nll_bic['model_name'], nll_bic['bic'], hue=nll_bic['color'], dodge=False)
ax.get_legend().remove()
plt.xticks(rotation='vertical')
plt.ylabel('BIC')
plt.ylim(0, 40000)
plt.savefig("{0}plot_bics.png".format(save_dir))

# Plot AICs of all models
nll_bic = nll_bic.sort_values(by=['aic'])
plt.figure(figsize=(15, 10))
ax = sns.barplot(nll_bic['model_name'], nll_bic['aic'], hue=nll_bic['color'], dodge=False)
ax.get_legend().remove()
plt.xticks(rotation='vertical')
plt.ylabel('BIC')
plt.ylim(0, 40000)
plt.savefig("{0}plot_aics.png".format(save_dir))

# Plot NLLs of all models
nll_bic = nll_bic.sort_values(by=['nll'])
plt.figure(figsize=(15, 10))
ax = sns.barplot(nll_bic['model_name'], nll_bic['nll'], hue=nll_bic['color'], dodge=False)
ax.get_legend().remove()
plt.xticks(rotation='vertical')
plt.ylabel('NLL')
plt.ylim(0, 20000)
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

        # Plot fitted parameter distributions
        try:
            param_names = [key for key in map.keys() if '__' not in key]
            n_subplots = len(param_names)
            plt.figure(figsize=(5, 2 * n_subplots))
            for i, param_name in enumerate(param_names):
                plt.subplot(n_subplots, 1, i + 1)
                # plt.hist(map[key])
                # sns.kdeplot(map[key])
                sns.distplot(map[param_name])
                plt.title(param_name)
            plt.tight_layout()
            print("Plotting fitted parameter distributions for model {0} to {1}.".format(file_name, save_dir))
            plt.savefig("{0}fitted_param_distr_{1}.png".format(save_dir, file_name[:-7]))
        except:
            print("Couldn't plot {0}.".format(file_name[:-7]))
#
# # Plot fitted against simulated nll (run after PSAllSimulations)
# nll = pd.read_csv(save_dir + 'nlls.csv')
# nll_all = nll.merge(nll_bic)
# ax = sns.scatterplot('nll', 'simulated_nll', hue='color', data=nll_all)
# for row in range(nll_all.shape[0]):
#     ax.text(nll_all.nll[row]+0.2, nll_all.simulated_nll[row], nll_all.model_name[row],
#             horizontalalignment='left', size='small', color='black')
# ax.get_legend().remove()
# plt.xlabel('Fitted NLL')
# plt.ylabel('Simulated NLL')
# plt.savefig("{0}plot_nll_sim_rec.png".format(save_dir))
