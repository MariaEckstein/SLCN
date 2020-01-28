import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import numpy as np
import os
import pandas as pd
import pymc3 as pm
import pickle
import seaborn as sns
sns.set(style='whitegrid')

from PSModelFunctions2 import get_paths, get_n_params
import scipy.stats as stats


class AnalyzePSModels:

    def __init__(self, data):

        # Basics (get from data dictionary)
        self.save_dir = data['save_dir']
        self.SLCN_info_file_dir = data['SLCN_info_file_dir']
        self.n_trials = data['n_trials']
        self.make_csvs_from_pickle = data['make_csvs_from_pickle']
        self.make_analyses = data['make_analyses']
        self.make_traceplot = data['make_traceplot']
        self.make_plots = data['make_plots']
        self.waic_criterion_for_analysis = data['waic_criterion_for_analysis']
        self.fit_mcmc = data['fit_mcmc']
        self.run_on_humans = data['run_on_humans']

        # Stuff I make myself in the later methods
        self.modelwise_LLs = []
        self.subjwise_LLs = []
        self.initialize()
        if not os.path.exists(self.save_dir + 'plots/'):
            os.makedirs(self.save_dir + 'plots/')
        self.SLCN_info = pd.read_csv(self.SLCN_info_file_dir)
        self.SLCN_info = self.SLCN_info.rename(columns={'ID': 'sID'})

    def initialize(self):
        self.file_name = ''

        self.fitted_params = []
        self.fitted_params_g = []
        self.n_subj = 0
        self.sID = []
        self.summary = None

        self.model_name = ''
        self.slope_variable = ''
        self.param_names = []
        self.group_level_param_names = []
        self.indiv_param_names = []
        self.NLL_param_names = []

        self.corrs = pd.DataFrame()

    def pickle2csv(self, file_names):

        # Get list of pickle files; initialize list for WAICs
        for file_name in file_names:
            self.initialize()
            print("Unpickling file {0} in {1}".format(file_name, self.save_dir))

            # Find out if this is map or mcmc data
            self.model_name = [part for part in file_name.split('_') if ('RL' in part) or ('B' in part) or ('WSLS' in part)][0]

            # Load pickled model fitting results
            print("\nLoading {0} from {1}".format(file_name, self.save_dir))
            with open(self.save_dir + file_name, 'rb') as handle:
                data = pickle.load(handle)

            # Unpack it
            self.sID = data['sIDs']
            self.n_subj = len(self.sID)
            self.slope_variable = data['slope_variable']
            if self.fit_mcmc:
                self.summary = data['summary'].T
                pd.DataFrame(self.summary).to_csv(self.get_file_name('summary'))
                waic = data['WAIC']
                trace = data['trace']
                samples_0 = data['trace']._straces[0].samples  # chain 0
                samples_1 = data['trace']._straces[1].samples  # chain 1
                samples_all = {key: list(samples_0[key]) + list(samples_1[key]) for key in samples_0.keys()}
                nll = 0
            else:
                self.summary = data['map']
                waic = data['aic']
                nll = data['nll']

            # Make traceplot
            if self.fit_mcmc and self.make_traceplot:
                pm.traceplot(trace)
                plt.savefig(self.get_file_name('traceplot'))

            # Get model_name and param_names
            self.get_param_names_from_summary()

            # Get p-values
            slope_params = []
            if self.fit_mcmc:
                if self.indiv_param_names[0] + '_slope' in samples_all.keys():
                    slope_params = [name + '_slope' for name in self.indiv_param_names]
                if self.indiv_param_names[0] + '_2_slope' in samples_all.keys():
                    slope_params += [name + '_2_slope' for name in self.indiv_param_names]

            p_values = pd.DataFrame()
            for slope_param in slope_params:
                row = pd.DataFrame(data=[np.mean(np.array(samples_all[slope_param]) > 0)], index=[slope_param], columns=['p'])
                p_values = p_values.append(row)
            p_values.to_csv(self.get_file_name('p_values'))

            # Add model WAIC score
            self.modelwise_LLs.append([self.model_name, self.slope_variable, self.n_subj, waic, nll])
            pd.DataFrame(self.modelwise_LLs).to_csv(self.get_file_name('modelwise_LLs', '_temp'), index=False)

            if self.fit_mcmc:
                columns = [key for key in self.summary.keys() if 'trialwise_LLs' in key]
                trialwise_LLs = self.summary.loc['mean', columns]
            else:
                trialwise_LLs = pd.DataFrame(self.summary['trialwise_LLs'], columns=list(self.sID))
                trialwise_LLs.to_csv(self.get_file_name('trialwise_LLs'), index=False)

                self.n_params = get_n_params(self.model_name, n_subj=1, n_groups=1)
                self.subjwise_LLs.append([self.model_name, self.slope_variable, self.n_subj, self.n_params, *np.sum(trialwise_LLs, axis=0)])
                pd.DataFrame(self.subjwise_LLs).to_csv(self.get_file_name('subjwise_LLs', '_temp'), index=False)

            # Create csv file for individuals' fitted parameters
            for param_name in self.indiv_param_names:
                if not self.fit_mcmc:
                    self.fitted_params.append(self.summary[param_name])  # TODO seems like calpha reappears in calpha_sc plots?
                else:
                    self.fitted_params.append(list(self.summary.loc['mean', [param_name + '__' + str(i) for i in range(self.n_subj)]]))
            self.fitted_params = np.array(self.fitted_params).T
            self.fitted_params = pd.DataFrame(self.fitted_params, columns=self.indiv_param_names)
            self.fitted_params['sID'] = list(self.sID)
            self.fitted_params['slope_variable'] = self.slope_variable

            # Add remaining info (PDS, T1, PreciseYrs, etc.)
            if self.run_on_humans:
                self.fitted_params = self.fitted_params.merge(self.SLCN_info, on='sID')  # columns only added for rows that already exist
                self.fitted_params.loc[self.fitted_params['Gender'] == 1, 'Gender'] = 'Male'
                self.fitted_params.loc[self.fitted_params['Gender'] == 2, 'Gender'] = 'Female'

                # Add '_quant' columns for age, PDS, T1
                if np.any(self.fitted_params['PreciseYrs'] < 18):
                    self.get_quantile_groups('PreciseYrs')
                    self.get_quantile_groups('PDS')
                    self.get_quantile_groups('T1')
            else:
                for col in ['Gender', 'PreciseYrs', 'PDS', 'T1', 'age_quant', 'PDS_quant', 'T1_quant']:
                    self.fitted_params[col] = 0

            # Add missing (non-fitted) parameters and save
            self.fill_in_missing_params()
            self.fitted_params['model'] = self.model_name
            self.fitted_params.to_csv(self.get_file_name('fitted_params'), index=False)

            # Create csv for group-level fitted parameters
            if self.group_level_param_names:
                if self.fit_mcmc:
                    self.fitted_params_g = self.summary[self.group_level_param_names]
                else:
                    self.fitted_params_g = []
                    for param_name in self.group_level_param_names:
                        self.fitted_params_g.append(self.summary[param_name])
                    self.fitted_params_g = np.array(self.fitted_params_g).T
                    self.fitted_params_g = pd.DataFrame(self.fitted_params_g,
                                                   columns=self.group_level_param_names)  # TODO , index=['Male', 'Female'])
                    self.fitted_params_g['slope_variable'] = self.slope_variable
                self.fitted_params_g.to_csv(self.get_file_name('fitted_params_g'))

        # Save modelwise LLs
        self.modelwise_LLs = pd.DataFrame(self.modelwise_LLs, columns=['model_name', 'slope_variable', 'n_subj', 'WAIC', 'NLL'])
        self.modelwise_LLs['AIC'] = self.modelwise_LLs['WAIC']
        self.modelwise_LLs.to_csv(self.get_file_name('modelwise_LLs'), index=False)

        self.subjwise_LLs = pd.DataFrame(self.subjwise_LLs, columns=['model_name', 'slope_variable', 'n_subj', 'n_params', *self.sID])
        self.subjwise_LLs.to_csv(self.get_file_name('subjwise_LLs'), index=False)

    def get_file_name(self, file_name, file_name_apx=''):

        if file_name == 'fitted_params_g':
            return '{0}params_g_{1}_{2}_{3}_pymc3.csv'.format(self.save_dir, self.model_name, self.slope_variable, self.n_subj)
        elif file_name == 'fitted_params':
            return '{0}params_{1}_{2}_{3}_pymc3.csv'.format(self.save_dir, self.model_name, self.slope_variable, self.n_subj)
        elif file_name == 'summary':
            return '{0}plots/summary_{1}_{2}_{3}_pymc3.csv'.format(self.save_dir, self.model_name, self.slope_variable, self.n_subj)
        elif file_name == 'param_plot':
            return "{0}plots/fitted_param_{4}_{1}_{2}_{3}.png".format(self.save_dir, self.model_name, self.slope_variable, self.n_subj, file_name_apx)
        elif file_name == 'param_corrs':
            return "{0}plots/corrs_{1}_{2}_{3}.csv".format(self.save_dir, self.model_name, self.slope_variable, self.n_subj)
        elif file_name == 'param_pairplot':
            return "{0}plots/param_pairplot_{1}_{2}_{3}.png".format(self.save_dir, self.model_name, self.slope_variable, self.n_subj)
        elif file_name == 'modelwise_LLs':
            return '{0}plots/modelwise_LLs{1}.csv'.format(self.save_dir, file_name_apx)
        elif file_name == 'traceplot':
            return "{0}plots/traceplot_{1}_{2}_{3}.png".format(self.save_dir, self.model_name, self.slope_variable, self.n_subj)
        elif file_name == 'AIC_plot':
            return "{0}plots/modelwise_AICs.png".format(self.save_dir)
        elif file_name == 'trialwise_LLs':
            return "{0}plots/trialwise_LLs_{1}_{2}_{3}_{4}.csv".format(self.save_dir, self.model_name, self.slope_variable, self.n_subj, file_name)
        elif file_name == 'trialwise_LLs_heatmap':
            return "{0}plots/trialwise_LLs_{1}_{2}_{3}_{4}.png".format(self.save_dir, self.model_name, self.slope_variable, self.n_subj, file_name)
        elif file_name == 'subjwise_LLs':
            return "{0}plots/subjwise_LLs{1}.csv".format(self.save_dir, file_name_apx)
        elif file_name == 'p_values':
            return "{0}plots/p_values_{1}_{2}_{3}.csv".format(self.save_dir, self.model_name, self.slope_variable, self.n_subj)
        else:
            raise ValueError("File can be 'fitted_params_g', 'fitted_params', or 'summary'. Fix get_file_name().")

    def get_info_from_filename(self, file_name):
        components = file_name.split('_')
        return components[1], int(components[-2])

    def z_score(self, vector):
        return (vector - np.mean(vector)) / np.std(vector)

    def analyze(self):

        # Plot AIC scores
        if not self.fit_mcmc:
            self.subjwise_LLs = pd.read_csv(self.get_file_name('subjwise_LLs'))
            subjwise_LLs = self.subjwise_LLs.drop(['slope_variable', 'n_subj'], axis=1)
            subjwise_LLs = pd.melt(subjwise_LLs, id_vars=['model_name', 'n_params'], value_vars=subjwise_LLs.columns[2:],
                                   value_name='NLL', var_name='sID')
            subjwise_LLs['BIC'] = np.log(self.n_trials) * subjwise_LLs['n_params'] - 2 * subjwise_LLs['NLL']
            subjwise_LLs['AIC'] = 2 * subjwise_LLs['n_params'] - 2 * subjwise_LLs['NLL']
            subjwise_LLs = subjwise_LLs.sort_values(by=['AIC'])
            plt.figure(figsize=(5, 5))
            sns.barplot(x='model_name', y='AIC', data=subjwise_LLs)
            plt.xticks(rotation='vertical')
            plt.ylabel('AIC')
            plt.savefig(self.get_file_name('AIC_plot'))
        else:
            try:
                self.modelwise_LLs = pd.read_csv(self.get_file_name('modelwise_LLs', ''))
                self.modelwise_LLs['ext_model_name'] = self.modelwise_LLs['slope_variable'] + '_' + [
                    str(i) for i in self.modelwise_LLs['n_subj']] + '_' + self.modelwise_LLs['model_name']
                self.modelwise_LLs = self.modelwise_LLs.sort_values(by=['AIC'])
                self.modelwise_LLs = self.modelwise_LLs[self.modelwise_LLs['AIC'] > 0]
                plt.figure(figsize=(5, 5))
                sns.barplot(x='ext_model_name', y='AIC', hue=self.modelwise_LLs['model_name'].str[0], data=self.modelwise_LLs, dodge=False)
                plt.xticks(rotation='vertical')
                plt.ylabel('AIC')
                plt.savefig(self.get_file_name('AIC_plot'))
            except FileNotFoundError:
                pass

        # Read in the new csv files and analyze them
        file_names = [f for f in os.listdir(self.save_dir) if
                      ('.csv' in f) and ('params' in f) and ('params_g' not in f)]

        for file_name in file_names:
            self.initialize()
            self.file_name = file_name
            print("Analyzing file {0} in {1}".format(file_name, self.save_dir))

            self.fitted_params = pd.read_csv(self.save_dir + file_name)
            self.fitted_params['age'] = self.fitted_params['PreciseYrs']
            self.fitted_params['age_z'] = self.z_score(self.fitted_params['age'])
            self.fitted_params['PDS_z'] = self.z_score(self.fitted_params['PDS'])
            self.fitted_params['T1_log'] = np.log(self.fitted_params['T1'])
            self.fitted_params['T1_log_z'] = self.z_score(self.fitted_params['T1_log'])

            try:
                self.fitted_params['T1_log_quant'] = np.log(self.fitted_params['T1_quant'])
                self.fitted_params['age_quant'] = self.fitted_params['PreciseYrs_quant']
            except KeyError:
                pass

            self.slope_variable = self.fitted_params['slope_variable'][0]
            self.sID = self.fitted_params['sID']
            self.model_name, self.n_subj = self.get_info_from_filename(file_name)

            try:
                self.fitted_params_g = pd.read_csv(self.get_file_name('fitted_params_g'), index_col=0)  # TODO comment back in!
                # self.fitted_params_g = pd.read_csv(self.save_dir + '/params_g_Bbsprywtv_age_z_291_pymc3.csv', index_col=0)
                # self.fitted_params_g = pd.read_csv(self.save_dir + '/params_g_Bbsprywtv_age_z_271_pymc3.csv', index_col=0)
                # self.fitted_params_g = pd.read_csv('C:/Users/maria/MEGAsync/SLCN/PShumanData/fitting/map_indiv/new_ML_models/MCMC/clustermodels/params_g_Bbsprywtv_age_z_271_pymc3.csv', index_col=0)
            except FileNotFoundError:
                self.fitted_params_g = pd.DataFrame()

            if self.fit_mcmc:
                self.summary = pd.read_csv(self.get_file_name('summary'), index_col=0)  # TODO comment back in!
                # self.summary = pd.read_csv(self.save_dir + '/plots/summary_Bbsprywtv_age_z_291_pymc3.csv', index_col=0)
                # self.summary = pd.read_csv(self.save_dir + '/g/summary_Bbsprytv_age_z_271_pymc3.csv', index_col=0)
                # self.summary = pd.read_csv('C:/Users/maria/MEGAsync/SLCN/PShumanData/fitting/map_indiv/new_ML_models/MCMC/clustermodels/plots/summary_Bbsprywtv_age_z_271_pymc3.csv', index_col=0)
                self.get_param_names_from_summary()
            if 'RL' in self.file_name:
                self.fitted_params['alpha_minus_calpha'] = self.fitted_params['alpha'] - self.fitted_params['calpha']
                self.fitted_params['alpha_minus_nalpha'] = self.fitted_params['alpha'] - self.fitted_params['nalpha']
                self.fitted_params['nalpha_minus_cnalpha'] = self.fitted_params['nalpha'] - self.fitted_params['cnalpha']
                self.indiv_param_names = ['beta', 'persev', 'alpha', 'nalpha']#, 'calpha', 'cnalpha',
                                          # 'alpha_minus_calpha', 'alpha_minus_nalpha', 'nalpha_minus_cnalpha']
            elif 'B' in self.file_name:
                self.indiv_param_names = ['p_reward', 'p_switch', 'beta', 'persev']

            # Close all figures
            plt.close('all')

            # Plot correlations between parameters and age / PDS / T1
            if self.make_plots:
                cols = ['age', 'PDS', 'T1_log']
                self.plot_fitted_params_against_cols(cols=[col + '_z' for col in cols], file_name_apx='distr', type='scatter')
                if np.any(self.fitted_params['PreciseYrs'] < 20):
                    self.plot_fitted_params_against_cols(
                        cols=[col + '_quant' for col in cols], file_name_apx='quant', type='line')

                # Plot correlations between and distributions of parameters
                sns.pairplot(self.fitted_params, hue='Gender', vars=self.indiv_param_names, plot_kws={'s': 10})
                plt.savefig(self.get_file_name('param_pairplot'))

                # Plot trialwise LLs
                try:
                    self.trialwise_LLs = pd.read_csv(self.get_file_name('trialwise_LLs'))
                    plt.figure(figsize=(40, 20))
                    sns.heatmap(self.trialwise_LLs)
                    plt.savefig(self.get_file_name('trialwise_LLs_heatmap'))
                except FileNotFoundError:
                    pass

        # self.modelwise_LLs = pd.read_csv(self.save_dir + 'plots/modelwise_LLs.csv')
        # self.modelwise_LLs = self.modelwise_LLs[(self.modelwise_LLs['model_name'].isin(['WSLS', 'WSLSS', 'RLab', 'RLabc', 'RLabcp', 'RLabcpn', 'Bb', 'Bbs', 'Bbsp', 'Bbspr'])) & (self.modelwise_LLs['n_subj'] == 160)]
        # self.modelwise_LLs['ext_model_name'] = self.modelwise_LLs['slope_variable'] + '_' + [str(i) for i in self.modelwise_LLs['n_subj']] + '_' + self.modelwise_LLs['model_name']
        # self.modelwise_LLs = self.modelwise_LLs.sort_values(by=['WAIC'])
        # self.modelwise_LLs = self.modelwise_LLs[self.modelwise_LLs['AIC'] > 0]
        # plt.figure(figsize=(5, 5))
        # ax = sns.barplot(x='ext_model_name', y='AIC', hue=self.modelwise_LLs['self.model_name'].str[0], data=self.modelwise_LLs, dodge=False)
        # plt.xticks(rotation='vertical')
        # plt.ylabel('AIC')
        # plt.ylim((0, 20000))
        # plt.savefig("{0}plots/all_poster_AICs.svg".format(self.save_dir))

        # Plot calculated (simulation code) and fitted (pymc3) NLLs against each other
        try:
            modelwise_LLs_sim = pd.read_csv(self.save_dir + 'plots/modelwise_LLs_sim.csv')
            self.modelwise_LLs = self.modelwise_LLs.merge(modelwise_LLs_sim)
            plt.figure(figsize=(10, 10))
            sns.scatterplot(x='NLL', y='fitted_nll', hue=self.modelwise_LLs['model_name'].str[0], data=self.modelwise_LLs)
            for i, type in enumerate(self.modelwise_LLs['model_name']):
                plt.text(self.modelwise_LLs['NLL'][i], self.modelwise_LLs['fitted_nll'][i], self.modelwise_LLs['model_name'][i],
                         fontsize=7)
            min_val, max_val = np.min(self.modelwise_LLs['fitted_nll']), np.max(self.modelwise_LLs['fitted_nll'])
            plt.plot([min_val, max_val], [min_val, max_val], c='grey')
            plt.xlabel('pymc3')
            plt.ylabel('sim')
            plt.savefig(self.save_dir + "plots/plot_NLLs.png")
        except FileNotFoundError:
            pass

    def get_quantile_groups(self, col):

        qcol = col + '_quant'
        self.fitted_params[qcol] = np.nan

        # Set adult quantile to 2
        self.fitted_params.loc[self.fitted_params['PreciseYrs'] > 20, qcol] = 2

        # Determine quantiles, separately for both genders
        for gender in np.unique(self.fitted_params['Gender']):

            # Get 4 quantiles
            cut_off_values = np.nanquantile(
                self.fitted_params.loc[(self.fitted_params.PreciseYrs < 20) & (self.fitted_params.Gender == gender), col],
                [0, 1/4, 2/4, 3/4])
            for cut_off_value, quantile in zip(cut_off_values, np.round([1/4, 2/4, 3/4, 1], 2)):
                self.fitted_params.loc[
                    (self.fitted_params.PreciseYrs < 20) & (self.fitted_params[col] >= cut_off_value) & (self.fitted_params.Gender == gender),
                    qcol] = quantile

    def get_param_names_from_summary(self):

        if self.fit_mcmc:
            self.group_level_param_names = [key for key in self.summary.keys() if ('__' not in key) or ('sd' in key) or ('int' in key) or ('slope' in key)]
            if 'a' in self.model_name:
                self.indiv_param_names.append('alpha')
            if ('b' in self.model_name) or ('WSLS' in self.model_name):
                self.indiv_param_names.append('beta')
            if 'c' in self.model_name:
                self.indiv_param_names.append('calpha')
            if 'n' in self.model_name:
                self.indiv_param_names.append('nalpha')
            if 'x' in self.model_name:
                self.indiv_param_names.append('cnalpha')
            if 'p' in self.model_name:
                self.indiv_param_names.append('persev')
            if 'd' in self.model_name:
                self.indiv_param_names.append('bias')
            if 's' in self.model_name:
                self.indiv_param_names.append('p_switch')
            if 'r' in self.model_name:
                self.indiv_param_names.append('p_reward')
            self.param_names = self.indiv_param_names + self.group_level_param_names

        else:
            self.param_names = [key for key in self.summary.keys() if ('__' not in key) or ('sd' in key) or ('slope' in key) or (('int' in key) and ('interval' not in key))]
            self.group_level_param_names = [name for name in self.param_names if ('int' in name) or ('slope' in name) or ('sd' in name)]
            self.NLL_param_names = [name for name in self.param_names if ('wise' in name)]
            self.indiv_param_names = [name for name in self.param_names if (name not in self.group_level_param_names) and (name not in self.NLL_param_names)]

    def fill_in_missing_params(self):

        if ('b' not in self.model_name) and ('WSLS' not in self.model_name):
            self.fitted_params['beta'] = 1
        if 'p' not in self.model_name:
            self.fitted_params['persev'] = 0

        if 'RL' in self.model_name:
            if 'a' not in self.model_name:
                self.fitted_params['alpha'] = 1
            if 'n' not in self.model_name:
                self.fitted_params['nalpha'] = self.fitted_params['alpha']
            if 'c' not in self.model_name:
                if '2' in self.model_name:  # RLabcnp2 model (alpha == calpha; nalpha == cnalpha)
                    self.fitted_params['calpha'] = self.fitted_params['alpha']
                else:  # RLabcnp model
                    self.fitted_params['calpha'] = 0
            if 'x' not in self.model_name:
                if '2' in self.model_name:
                    self.fitted_params['cnalpha'] = self.fitted_params['nalpha']
                else:
                    self.fitted_params['cnalpha'] = 0

        if 'B' in self.model_name:
            self.fitted_params['p_noisy'] = 1e-5
            if 's' not in self.model_name:
                self.fitted_params['p_switch'] = 0.05081582
            if 'r' not in self.model_name:
                self.fitted_params['p_reward'] = 0.75

    def get_slope_letter_for_param_name(self, param_name):
        if param_name == 'alpha':
            return 'l'
        elif param_name == 'nalpha':
            return 'q'
        elif param_name == 'calpha':
            return 'o'
        elif param_name == 'cnalpha':
            return 'u'
        elif param_name == 'm':
            return 'z'
        elif param_name == 'p_switch':
            return 'w'
        elif param_name == 'p_reward':
            return 'v'
        elif param_name == 'beta':
            return 'y'
        elif param_name == 'persev':
            return 't'
        else:
            return 'dddddddd'

    def plot_fitted_params_against_cols(self, cols=('age_z', 'PDS_z', 'T1_log_z'), file_name_apx='', type='scatter'):

        self.corrs = pd.DataFrame()

        n_rows = max(len(self.indiv_param_names), 2)  # need at least 2 rows for indexing to work
        n_cols = max(len(cols), 2)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.1 * n_cols, 2 * n_rows), sharex='col', sharey='row')

        for row, param_name in enumerate(self.indiv_param_names):
            for col, pred in enumerate(cols):

                # Plot correlations
                if type == 'scatter':

                    axes[row, col].scatter(self.fitted_params[pred], self.fitted_params[param_name], s=1,
                                           c=['tomato' if g == 'Female' else 'lightseagreen' for g in self.fitted_params['Gender']])

                    # If this model has a fitted slope, add it to the plot
                    right_param = self.get_slope_letter_for_param_name(param_name) in self.model_name
                    if right_param and (pred in self.file_name):
                        slope = self.summary.filter(regex=param_name + '_slope').loc['mean'][0]
                        try:
                            slope2 = self.summary.filter(regex=param_name + '_2_slope').loc['mean'][0]
                        except:
                            slope2 = 0
                        sd = self.summary.filter(regex=param_name + '_sd').loc['mean'][0]
                        intercept = self.summary.filter(regex=param_name + '_int.*').loc['mean'][0]
                        x_vals = np.linspace(np.min(self.fitted_params[pred]), np.max(self.fitted_params[pred]), 100)
                        y_vals = intercept + slope * x_vals + slope2 * x_vals * x_vals
                        axes[row, col].plot(x_vals, y_vals, '--', c='black')
                        axes[row, col].plot(x_vals, y_vals + sd, '--', c='grey')
                        axes[row, col].plot(x_vals, y_vals - sd, '--', c='grey')

                elif type == 'line':
                    sns.lineplot(pred, param_name, hue='Gender', data=self.fitted_params, legend=False, ax=axes[row, col], palette={'Female': 'tomato', 'Male': 'lightseagreen'})

                # Make y-axis [0, 1] for all alphas
                if ('alpha' in param_name) and ('minus' not in param_name):
                    axes[row, col].set_ylim([0, 1])
                if 'beta' == param_name:
                    axes[row, col].set_ylim([1, 11])
                if 'persev' == param_name:
                    axes[row, col].set_ylim([-0.25, 0.75])

                # Add axis labels
                if row == n_rows - 1:  # bottom row
                    axes[row, col].set_xlabel(pred)
                if col == 0:  # left-most column
                    axes[row, col].set_ylabel(param_name)

                # Calculate correlations
                for gen in np.unique(self.fitted_params['Gender']):
                    gen_idx = self.fitted_params['Gender'] == gen

                    clean_idx = 1 - np.isnan(self.fitted_params[pred]) | np.isnan(self.fitted_params[param_name])
                    corr, p = stats.pearsonr(
                        self.fitted_params.loc[clean_idx & gen_idx, param_name], self.fitted_params.loc[clean_idx & gen_idx, pred])

                    clean_idx_young = clean_idx & (self.fitted_params['PreciseYrs'] < 20) & gen_idx
                    corr_young, p_young = stats.pearsonr(
                        self.fitted_params.loc[clean_idx_young, param_name], self.fitted_params.loc[clean_idx_young, pred])

                    self.corrs = self.corrs.append([[param_name, pred, gen, corr, p, corr_young, p_young]])

        # Finish plot
        plt.tight_layout()
        plt.savefig(self.get_file_name('param_plot', file_name_apx))

        # Save correlations
        self.corrs.columns = ['param_name', 'charact', 'gender', 'r', 'p', 'r_young', 'p_young']
        self.corrs.to_csv(self.get_file_name('param_corrs'), index=False)


# Main script
data = {
    # 'save_dir': 'C:/Users/maria/MEGAsync/SLCN/PShumanData/fitting/map_indiv/new_ML_models/MCMC/clustermodels/genrec/simRL/',  #/genrec/simBbspr/',
    # 'save_dir': 'C:/Users/maria/MEGAsync/SLCN/PShumanData/fitting/map_indiv/new_ML_models/',
    # 'SLCN_info_file_dir': 'C:/Users/maria/MEGAsync/SLCNdata/SLCNinfo2.csv',
    # 'n_trials': 120,
    'save_dir': 'C:/Users/maria/MEGAsync/SLCN/PShumanData/fitting/map_indiv/mice/',
    'SLCN_info_file_dir': 'C:/Users/maria/MEGAsync/SLCN/PSMouseData/age.csv',
    'n_trials': 780,
    'make_csvs_from_pickle': True,
    'make_analyses': True,
    'make_traceplot': False,
    'make_plots': True,
    'waic_criterion_for_analysis': 1e6,
    'fit_mcmc': False,
    'run_on_humans': True,
}

apm = AnalyzePSModels(data)

# Read all pickle files in save_dir and create csv files fitted_params and fitted_params_g
if apm.make_csvs_from_pickle:
    file_names = [f for f in os.listdir(apm.save_dir) if '.pickle' in f]
    apm.pickle2csv(file_names)

# Create files, save files, make plots, save plots
if apm.make_analyses:
    apm.analyze()
