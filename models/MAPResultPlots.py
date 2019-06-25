import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pymc3 as pm
import pickle
import seaborn as sns
sns.set(style='whitegrid')

from shared_modeling_simulation import get_paths, get_n_params
import scipy.stats as stats


save_dir = 'C:/Users/maria/MEGAsync/SLCN/PShumanData/fitting/map_indiv/new_map_models_Bayes_mcmc/'
SLCN_info_file_dir = 'C:/Users/maria/MEGAsync/SLCNdata/SLCNinfo2.csv'
n_subj = 234  # all: 234  # just kids & teens: 160
n_trials = 120
make_traceplot = True
make_plots = True
# ages = pd.read_csv(get_paths(False)['ages_cluster'])  # TODO model fitted on cluster or laptop??? Orders differ!!!
ages = pd.read_csv(get_paths(False)['ages'])  # TODO model fitted on cluster or laptop??? Orders differ!!!
SLCN_info = pd.read_csv(SLCN_info_file_dir)
SLCN_info = SLCN_info.rename(columns={'ID': 'sID'})
waic_criterion_for_analysis = 1e6


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


def get_param_names_from_model_name(model_name, fit_mcmc, fit_map, summary=None):

    if fit_mcmc:
        group_level_param_names = [key for key in summary.keys() if ('__' not in key) or ('sd' in key) or ('int' in key) or ('slope' in key)]
        indiv_param_names = []
        NLL_param_names = []
        if 'a' in model_name:
            indiv_param_names.append('alpha')
        if ('b' in model_name) or ('WSLS' in model_name):
            indiv_param_names.append('beta')
        if 'c' in model_name:
            indiv_param_names.append('calpha')
        if 'n' in model_name:
            indiv_param_names.append('nalpha')
        if 'x' in model_name:
            indiv_param_names.append('cnalpha')
        if 'p' in model_name:
            indiv_param_names.append('persev')
        if 's' in model_name:
            indiv_param_names.append('p_switch')
        if 'r' in model_name:
            indiv_param_names.append('p_reward')
        param_names = indiv_param_names + group_level_param_names

    elif fit_map:
        param_names = [key for key in summary.keys() if ('__' not in key) or ('sd' in key) or ('slope' in key) or (('int' in key) and ('interval' not in key))]
        group_level_param_names = [name for name in param_names if ('int' in name) or ('slope' in name) or ('sd' in name)]
        NLL_param_names = [name for name in param_names if ('wise' in name)]
        indiv_param_names = [name for name in param_names if (name not in group_level_param_names) and (name not in NLL_param_names)]

    return param_names, group_level_param_names, indiv_param_names, NLL_param_names


def fill_in_missing_params(fitted_params, model_name):

    if ('b' not in model_name) and ('WSLS' not in model_name):
        fitted_params['beta'] = 1
    if 'p' not in model_name:
        fitted_params['persev'] = 0

    if 'RL' in model_name:
        if 'n' not in model_name:
            fitted_params['nalpha'] = fitted_params['alpha']
        if 'c' not in model_name:
            fitted_params['calpha_sc'] = 0
            fitted_params['calpha'] = 0
        if 'x' not in model_name:
            fitted_params['cnalpha_sc'] = fitted_params['calpha_sc']
            fitted_params['cnalpha'] = fitted_params['cnalpha_sc'] * fitted_params['nalpha']

    if 'B' in model_name:
        if 's' not in model_name:
            fitted_params['p_switch'] = 0.05081582
        if 'r' not in model_name:
            fitted_params['p_reward'] = 0.75

    return fitted_params


def plot_fitted_params_against_cols(fitted_params, param_names, cols=['age', 'PDS', 'T1'], type='scatter'):

    corrs = pd.DataFrame()
    n_rows = max(len(param_names), 2)  # need at least 2 rows to make later indexing work
    n_cols = max(len(cols), 2)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows), sharex='col', sharey='row')

    for row, param_name in enumerate(param_names):
        for col, pred in enumerate(cols):

            # Plot correlations
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

            # Calculate correlations
            for gen in np.unique(fitted_params['Gender']):
                gen_idx = fitted_params['Gender'] == gen

                clean_idx = 1 - np.isnan(fitted_params[pred]) | np.isnan(fitted_params[param_name])
                corr, p = stats.pearsonr(
                    fitted_params.loc[clean_idx & gen_idx, param_name], fitted_params.loc[clean_idx & gen_idx, pred])

                clean_idx_young = clean_idx & (fitted_params['PreciseYrs'] < 20) & gen_idx
                corr_young, p_young = stats.pearsonr(
                    fitted_params.loc[clean_idx_young, param_name], fitted_params.loc[clean_idx_young, pred])

                corrs = corrs.append([[param_name, pred, gen, corr, p, corr_young, p_young]])

    # Finish plot
    plt.tight_layout()
    print("Plotting fitted parameters against {1} for model {0}.".format(file_name, cols))

    # Return corrs
    corrs.columns = ['param_name', 'charact', 'gender', 'r', 'p', 'r_young', 'p_young']
    return corrs


# Main analysis script
file_names = [f for f in os.listdir(save_dir) if '.pickle' in f]
modelwise_LLs = []

for file_name in file_names:

    # Find out if this is map or mcmc data
    if 'map' in file_name:
        fit_map = True
        fit_mcmc = False
    elif 'mcmc' in file_name:
        fit_mcmc = True
        fit_map = False

    # Load pickled model fitting results
    with open(save_dir + file_name, 'rb') as handle:
        print("\nLoading pickle file {0} from {1}".format(file_name, save_dir))
        data = pickle.load(handle)

    # Unpack it
    sID = data['sIDs']
    slope_variable = data['slope_variable']
    if fit_mcmc:
        summary = data['summary'].T
        waic = data['WAIC']
        trace = data['trace']
        nll = 0
    elif fit_map:
        summary = data['map']
        waic = data['aic']
        nll = data['nll']

    # Make traceplot
    if fit_mcmc and make_traceplot:
        pm.traceplot(trace)
        plt.savefig(save_dir + "plots/traceplot" + model_name + ".png")

    # Get model_name and param_names
    model_name = [part for part in file_name.split('_') if ('RL' in part) or ('B' in part) or ('WSLS' in part)][0]
    param_names, group_level_param_names, indiv_param_names, NLL_param_names = get_param_names_from_model_name(
        model_name, fit_mcmc, fit_map, summary)

    # Add model WAIC score
    # waics = waics.append([[model_name, slope_variable, waic, nll]])

    # Create csv file for individuals' fitted parameters
    fitted_params = []
    for param_name in indiv_param_names:
        if fit_map:
            fitted_params.append(summary[param_name])  # TODO seems like calpha reappears in calpha_sc plots?
        elif fit_mcmc:
            fitted_params.append(list(summary.loc['mean', [param_name + '__' + str(i) for i in range(n_subj)]]))
    fitted_params = np.array(fitted_params).T
    fitted_params = pd.DataFrame(fitted_params, columns=indiv_param_names)
    fitted_params['sID'] = list(sID)
    fitted_params['slope_variable'] = slope_variable

    # Add remaining info (PDS, T1, PreciseYrs, etc.)
    fitted_params = fitted_params.merge(SLCN_info, on='sID')  # columns are only added for rows that already exist
    fitted_params.loc[fitted_params['Gender'] == 1, 'Gender'] = 'Male'
    fitted_params.loc[fitted_params['Gender'] == 2, 'Gender'] = 'Female'

    # Add '_quant' columns for age, PDS, T1
    fitted_params = get_quantile_groups(fitted_params, 'PreciseYrs')
    fitted_params = get_quantile_groups(fitted_params, 'PDS')
    fitted_params = get_quantile_groups(fitted_params, 'T1')

    # Add missing (non-fitted) parameters and save
    fitted_params = fill_in_missing_params(fitted_params, model_name)
    print("Saving csv of fitted parameters for model {0} to {1}.".format(file_name, save_dir))
    fitted_params.to_csv(save_dir + 'params_' + model_name + '_pymc3.csv', index=False)

    # Create csv for group-level fitted parameters
    if group_level_param_names:
        if fit_mcmc:
            fitted_params_g = summary.loc['mean', group_level_param_names][:, None]
        elif fit_map:
            fitted_params_g = []
            for param_name in group_level_param_names:
                fitted_params_g.append(summary[param_name])
        fitted_params_g = np.array(fitted_params_g).T
        fitted_params_g = pd.DataFrame(fitted_params_g, columns=group_level_param_names)  # TODO , index=['Male', 'Female'])
        fitted_params_g['slope_variable'] = slope_variable
        fitted_params_g.to_csv(save_dir + 'params_g_' + model_name + '_pymc3.csv')

    if NLL_param_names:
        pd.DataFrame(summary['trialwise_LLs'], columns=sID).to_csv(save_dir + 'plots/trialwise_LLs_' + model_name + '_pymc3.csv', index=False)
        pd.DataFrame(summary['trialwise_p_right'], columns=sID).to_csv(save_dir + 'plots/trialwise_p_' + model_name + '_pymc3.csv', index=False)
        subjwise_LLs = pd.DataFrame(summary['subjwise_LLs'], index=sID, columns=['NLL_pymc3'])
        subjwise_LLs.to_csv(save_dir + 'plots/subjwise_LLs_' + model_name + '_pymc3.csv')

        n_params = get_n_params(model_name, n_subj, 1)
        nll = -np.sum(subjwise_LLs).values[0]
        bic = np.log(n_trials * n_subj) * n_params + 2 * nll
        aic = 2 * n_params + 2 * nll
        modelwise_LLs.append([nll, bic, aic, model_name])

    # Plot correlations between parameters and age/PDS/T1
    if make_plots:
        cols = ['PreciseYrs', 'PDS', 'T1']
        corrs = plot_fitted_params_against_cols(fitted_params, indiv_param_names, cols=cols, type='scatter')
        plt.savefig("{0}plots/fitted_param_distr_{1}_{2}.png".format(save_dir, model_name, slope_variable))
        plot_fitted_params_against_cols(fitted_params, indiv_param_names, cols=[col + '_quant' for col in cols], type='line')
        plt.savefig("{0}plots/fitted_param_quant_{1}_{2}.png".format(save_dir, model_name, slope_variable))

        # Save correlations
        print("Saving correlations between fitted parameters and age, PDS, T for model {0} to {1}.".format(file_name, save_dir))
        corrs.to_csv("{0}plots/corrs_{1}_{2}.csv".format(save_dir, model_name, slope_variable), index=False)

        # Plot correlations between and distributions of parameters
        sns.pairplot(fitted_params, hue='Gender', vars=indiv_param_names, plot_kws={'s': 10})
        print("Saving pairplot to {0}.".format(save_dir))
        plt.savefig("{0}plots/param_pairplot_{1}_{2}.png".format(save_dir, model_name, slope_variable))

    # Plot pymc3 NLLs (obtained when fitting) and my NLLs (using the simulation code)
    try:
        subjwise_LLs_sim = pd.read_csv(save_dir + 'plots/subjwise_LLs_' + model_name + '_sim.csv')
        subjwise_LLs = subjwise_LLs.merge(subjwise_LLs_sim, right_on='sID', left_index=True)
        plt.figure()
        sns.scatterplot(x='NLL_pymc3', y='NLL_sim', hue='sID', data=subjwise_LLs)  # , hue=subjwise_LLs['model_name'].str[0]
        min_val, max_val = np.min(subjwise_LLs['NLL_sim']), np.max(subjwise_LLs['NLL_sim'])
        plt.plot([min_val, max_val], [min_val, max_val], c='grey')
        plt.xlabel('pymc3')
        plt.ylabel('sim')
        plt.savefig(save_dir + "plots/plot_NLLs_" + model_name + ".png")
    except FileNotFoundError:
        pass

# Save modelwise LLs
modelwise_LLs = pd.DataFrame(modelwise_LLs, columns=['NLL', 'BIC', 'AIC', 'model_name'])
modelwise_LLs.to_csv(save_dir + 'plots/modelwise_LLs_pymc3.csv', index=False)

# Plot AIC scores
modelwise_LLs = modelwise_LLs.sort_values(by=['AIC'])
modelwise_LLs = modelwise_LLs[modelwise_LLs['AIC'] > 0]
plt.figure(figsize=(50, 10))
ax = sns.barplot(x='model_name', y='AIC', hue=modelwise_LLs['model_name'].str[0], data=modelwise_LLs, dodge=False)
plt.xticks(rotation='vertical')
plt.ylabel('AIC')
plt.savefig("{0}plots/modelwise_AICs.png".format(save_dir))

# Plot calculated (simulation code) and fitted (pymc3) NLLs against each other
try:
    modelwise_LLs_sim = pd.read_csv(save_dir + 'plots/modelwise_LLs_sim.csv')
    modelwise_LLs = modelwise_LLs.merge(modelwise_LLs_sim)
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x='NLL', y='fitted_nll', hue=modelwise_LLs['model_name'].str[0], data=modelwise_LLs)
    for i, type in enumerate(modelwise_LLs['model_name']):
        plt.text(modelwise_LLs['NLL'][i], modelwise_LLs['fitted_nll'][i], modelwise_LLs['model_name'][i], fontsize=7)
    min_val, max_val = np.min(modelwise_LLs['fitted_nll']), np.max(modelwise_LLs['fitted_nll'])
    plt.plot([min_val, max_val], [min_val, max_val], c='grey')
    plt.xlabel('pymc3')
    plt.ylabel('sim')
    plt.savefig(save_dir + "plots/plot_NLLs.png")
except FileNotFoundError:
    pass

# # Plot and save WAICS
# waics.columns = ['model_name', 'slope_variable', 'waic', 'nll']
# waics = waics.sort_values(by=['waic'])
# waics = waics[waics['waic'] > 0]
# plt.figure(figsize=(50, 10))
# ax = sns.barplot(waics['model_name'] + '_' + waics['slope_variable'].str.rstrip('_z'), waics['waic'])
# plt.xticks(rotation='vertical')
# plt.ylabel('AIC')
# plt.savefig("{0}waic_plot.png".format(save_dir))
# print("Saving waic scores for all models to {0}.".format(save_dir))
# waics.to_csv("{0}waics_pymc3.csv".format(save_dir), index=False)
#
# # Read in nll_bic
# if plot_nlls_bics_aics:
#     nll_bic = pd.read_csv(save_dir + 'nll_bics.csv')
#     nll_bic['color'] = nll_bic['model_name'].astype(str).str[:1]
#
#     if fit_map:
#         # Plot BICs of all models
#         nll_bic = nll_bic.sort_values(by=['bic'])
#         plt.figure(figsize=(25, 10))
#         ax = sns.barplot(nll_bic['model_name'], nll_bic['bic'], hue=nll_bic['color'], dodge=False)
#         ax.get_legend().remove()
#         plt.xticks(rotation='vertical')
#         plt.ylabel('BIC')
#         plt.ylim(0, 40000)
#         plt.savefig("{0}plot_bics.png".format(save_dir))
#
#         # Plot NLLs of all models
#         nll_bic = nll_bic.sort_values(by=['nll'])
#         plt.figure(figsize=(25, 10))
#         ax = sns.barplot(nll_bic['model_name'], nll_bic['nll'], hue=nll_bic['color'], dodge=False)
#         ax.get_legend().remove()
#         plt.xticks(rotation='vertical')
#         plt.ylabel('NLL')
#         plt.ylim(0, 20000)
#         plt.savefig("{0}plot_nlls.png".format(save_dir))
#
#     # Plot AICs of all models
#     nll_bic = nll_bic.sort_values(by=['aic'])
#     plt.figure(figsize=(25, 10))
#     ax = sns.barplot(nll_bic['model_name'], nll_bic['aic'], hue=nll_bic['color'], dodge=False)
#     ax.get_legend().remove()
#     plt.xticks(rotation='vertical')
#     plt.ylabel('AIC')
#     plt.ylim(0, 40000)
#     plt.savefig("{0}plot_aics.png".format(save_dir))

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
