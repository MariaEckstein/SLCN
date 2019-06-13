import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pymc3 as pm
import pickle
import seaborn as sns
sns.set(style='whitegrid')

from shared_modeling_simulation import get_paths
import scipy.stats as stats


save_dir = 'C:/Users/maria/MEGAsync/SLCN/PShumanData/fitting/map_indiv/cluster_fits/'
SLCN_info_file_dir = 'C:/Users/maria/MEGAsync/SLCNdata/SLCNinfo2.csv'
n_subj = 234  # TODO Will be 234 after 2019-06-12; was 233 before that
fit_map = False
fit_mcmc = True
make_traceplot = False
plot_nlls_bics_aics = False
ages = pd.read_csv(get_paths(False)['ages_cluster'])  # TODO model fitted on cluster or on laptop??? Orders differ!!!

# Read in nll_bic
if plot_nlls_bics_aics:
    nll_bic = pd.read_csv(save_dir + 'nll_bics.csv')
    nll_bic['color'] = nll_bic['model_name'].astype(str).str[:1]

    if fit_map:
        # Plot BICs of all models
        nll_bic = nll_bic.sort_values(by=['bic'])
        plt.figure(figsize=(25, 10))
        ax = sns.barplot(nll_bic['model_name'], nll_bic['bic'], hue=nll_bic['color'], dodge=False)
        ax.get_legend().remove()
        plt.xticks(rotation='vertical')
        plt.ylabel('BIC')
        plt.ylim(0, 40000)
        plt.savefig("{0}plot_bics.png".format(save_dir))

        # Plot NLLs of all models
        nll_bic = nll_bic.sort_values(by=['nll'])
        plt.figure(figsize=(25, 10))
        ax = sns.barplot(nll_bic['model_name'], nll_bic['nll'], hue=nll_bic['color'], dodge=False)
        ax.get_legend().remove()
        plt.xticks(rotation='vertical')
        plt.ylabel('NLL')
        plt.ylim(0, 20000)
        plt.savefig("{0}plot_nlls.png".format(save_dir))

    # Plot AICs of all models
    nll_bic = nll_bic.sort_values(by=['aic'])
    plt.figure(figsize=(25, 10))
    ax = sns.barplot(nll_bic['model_name'], nll_bic['aic'], hue=nll_bic['color'], dodge=False)
    ax.get_legend().remove()
    plt.xticks(rotation='vertical')
    plt.ylabel('AIC')
    plt.ylim(0, 40000)
    plt.savefig("{0}plot_aics.png".format(save_dir))


def get_quantile_groups(params_ages, col):

    qcol = col + '_quant'
    params_ages[qcol] = np.nan

    # Set adult quantile to 2
    params_ages.loc[params_ages['PreciseYrs'] > 20, qcol] = 2

    # Determine quantiles, separately for both genders
    for gender in np.unique(params_ages['Gender']):

        # Get 3 quantiles
        cut_off_values = np.nanquantile(
            params_ages.loc[(params_ages.PreciseYrs < 20) & (params_ages.Gender == gender), col],
            [0, 1 / 3, 2 / 3])
        for cut_off_value, quantile in zip(cut_off_values, np.round([1 / 3, 2 / 3, 1], 2)):
            params_ages.loc[
                (params_ages.PreciseYrs < 20) & (params_ages[col] >= cut_off_value) & (params_ages.Gender == gender),
                qcol] = quantile

    return params_ages


def get_param_names_from_model_name(model_name):
    param_names = []
    if 'a' in model_name:
        param_names.append('alpha')
    if 'b' in model_name:
        param_names.append('beta')
    if 'c' in model_name:
        param_names.append('calpha')
    if 'n' in model_name:
        param_names.append('nalpha')
    if 'x' in model_name:
        param_names.append('cnalpha')
    if 'p' in model_name:
        param_names.append('persev')
    if 's' in model_name:
        param_names.append('p_switch')
    if 'r' in model_name:
        param_names.append('p_reward')
    return param_names


def fill_in_missing_params(fitted_params, model_name):

    if 'b' not in model_name:
        fitted_params['beta'] = 1
    if 'p' not in model_name:
        fitted_params['persev'] = 0

    if 'RL' in model_name:
        if 'c' not in model_name:
            fitted_params['calpha'] = 0
        if 'n' not in model_name:
            fitted_params['nalpha'] = fitted_params['alpha']
        if 'x' not in model_name:
            fitted_params['cnalpha'] = fitted_params['calpha']

    if 'B' in model_name:
        if 's' not in model_name:
            fitted_params['p_switch'] = 0.05081582
        if 'r' not in model_name:
            fitted_params['p_reward'] = 0.75

    return fitted_params


def plot_fitted_params_against_cols(fitted_params, param_names, cols=['age', 'PDS', 'T1'], type='scatter'):
    n_rows = len(param_names)
    n_cols = len(cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows), sharex='col', sharey='row')
    for row, param_name in enumerate(param_names):
        for col, pred in enumerate(cols):
            if type == 'scatter':
                # sns.regplot(pred, param_name, data=fitted_params, ax=axes[row, col],
                #             scatter_kws={'s': 1, 'color': ['r' if g == 2 else 'b' for g in fitted_params['Gender']]})
                axes[row, col].scatter(fitted_params[pred], fitted_params[param_name], s=1,
                                       c=['r' if g == 'Female' else 'b' for g in fitted_params['Gender']])
            elif type == 'line':
                sns.lineplot(pred, param_name, hue='Gender', data=fitted_params, legend=False, ax=axes[row, col])
                # sns.barplot(pred, param_name, hue='Gender', data=fitted_params, ax=axes[row, col])  # , **{'tick_label':False}
            if row == n_rows - 1:
                axes[row, col].set_xlabel(pred)
            if col == 0:
                axes[row, col].set_ylabel(param_name)
    plt.tight_layout()
    print("Plotting fitted parameters against {1} for model {0}.".format(file_name, cols))


# Plot parameters against
if fit_mcmc:
    waics = pd.DataFrame()
    file_names = [f for f in os.listdir(save_dir) if '.pickle' in f]

    for file_name in file_names:
        model_name = [part for part in file_name.split('_') if ('RL' in part) or ('B' in part) or ('WSLS' in part)][0]
        param_names = get_param_names_from_model_name(model_name)

        # Load pickled model fitting results
        with open(save_dir + file_name, 'rb') as handle:
            print("\nLoading pickle file {0} from {1}...".format(file_name, save_dir))
            data = pickle.load(handle)

        # Unpack it
        summary = data['summary']
        waic = data['WAIC']
        trace = data['trace']

        # Add model WAIC score
        waics = waics.append([[model_name, waic]])

        # Make traceplot
        if make_traceplot:
            pm.traceplot(trace)
            plt.savefig("{0}traceplot{1}.png".format(save_dir, file_name[:-7]))

        # Create csv files with the fitted parameters for easier access
        fitted_params = ages.copy()
        for param_name in param_names:

            # Read in parameter in the order given by pymc3_idx, i.e., what was used for fitting
            param_values = summary.loc[[param_name + '__' + str(i) for i in fitted_params['pymc3_idx']], 'mean'].values
            fitted_params[param_name] = param_values

        # Add remaining info (PDS, T1, PreciseYrs, etc.)
        SLCN_info = pd.read_csv(SLCN_info_file_dir)
        SLCN_info = SLCN_info.rename(columns={'ID': 'sID'})

        fitted_params = fitted_params.merge(SLCN_info, on='sID')  # columns are only added for rows that already exist
        fitted_params.loc[fitted_params['Gender'] == 1, 'Gender'] = 'Male'
        fitted_params.loc[fitted_params['Gender'] == 2, 'Gender'] = 'Female'

        # Add '_quant' columns for age, PDS, T1
        fitted_params = get_quantile_groups(fitted_params, 'PreciseYrs')
        fitted_params = get_quantile_groups(fitted_params, 'PDS')
        fitted_params = get_quantile_groups(fitted_params, 'T1')

        # Add missing (non-fitted) parameters
        fitted_params = fill_in_missing_params(fitted_params, model_name)

        # Save fitted_params to csv
        print("Saving csv of fitted parameters for model {0} to {1}.".format(file_name, save_dir))
        fitted_params.to_csv(save_dir + model_name + '_fitted_params.csv', index=False)

        # Plot and save fitted parameter distributions
        cols = ['PreciseYrs', 'PDS', 'T1']
        plot_fitted_params_against_cols(fitted_params, param_names, cols=cols, type='scatter')
        plt.savefig("{0}fitted_param_distr_{1}.png".format(save_dir, model_name))
        plot_fitted_params_against_cols(fitted_params, param_names, cols=[col + '_quant' for col in cols], type='line')
        plt.savefig("{0}fitted_param_quant_{1}.png".format(save_dir, model_name))

        # Get correlations between parameters and age etc.
        corrs = pd.DataFrame()
        for param_name in param_names:
            for col in cols:
                for gen in np.unique(fitted_params['Gender']):

                    gen_idx = fitted_params['Gender'] == gen

                    clean_idx = 1 - np.isnan(fitted_params[col]) | np.isnan(fitted_params[param_name])
                    corr, p = stats.pearsonr(
                        fitted_params.loc[clean_idx & gen_idx, param_name], fitted_params.loc[clean_idx & gen_idx, col])

                    clean_idx_young = clean_idx & (fitted_params['PreciseYrs'] < 20) & gen_idx
                    corr_young, p_young = stats.pearsonr(
                        fitted_params.loc[clean_idx_young, param_name], fitted_params.loc[clean_idx_young, col])

                    corrs = corrs.append([[param_name, col, gen, corr, p, corr_young, p_young]])

        # Save correlations
        corrs.columns = ['param_name', 'charact', 'gender', 'r', 'p', 'r_young', 'p_young']
        print("Saving correlations between fitted parameters and age, PDS, T for model {0} to {1}.".format(file_name, save_dir))
        corrs.to_csv("{0}corrs_{1}.csv".format(save_dir, model_name), index=False)

    # Save WAICS
    print("Saving waic scores for all models to {0}.".format(save_dir))
    waics.columns = ['model_name', 'waic']
    waics.to_csv("{0}waics.csv".format(save_dir), index=False)


# Bring parameter estimates in the right shape
if fit_map:
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
