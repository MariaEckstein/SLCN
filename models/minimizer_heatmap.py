import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from itertools import combinations
import pickle


class CollectPaths(object):
    def __init__(self, colnames):
        self.all_points = pd.DataFrame(columns=colnames)
        self.row = 0

    def add_point(self, coordinates):
        self.all_points.loc[self.row, :] = coordinates
        self.row += 1

    def get(self):
        return self.all_points


class CollectMinima(object):
    def __init__(self, colnames):
        self.all_minima = pd.DataFrame(columns=colnames)
        self.row = 0

    def __call__(self, x, f, accept):
        self.all_minima.loc[self.row, :] = x
        self.row += 1

    def get(self):
        return self.all_minima


class PlotMinimizerHeatmap(object):
    def __init__(self, file_path):  # , heatmap_data, hoppin_minima, hoppin_paths, final_result
        self.file_path = file_path

    def pickle_brute_results(self, brute_results):
        with open(self.file_path + 'brute_results.pickle', 'wb') as handle:
            pickle.dump(brute_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def unpickle_brute_results(self):
        with open(self.file_path + 'brute_results.pickle', 'rb') as handle:
            return pickle.load(handle)

    def pickle_hoppin_results(self, hoppin_results):
        with open(self.file_path + 'hoppin_results.pickle', 'wb') as handle:
            pickle.dump(hoppin_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def unpickle_hoppin_results(self):
        with open(self.file_path + 'hoppin_results.pickle', 'rb') as handle:
            return pickle.load(handle)

    @staticmethod
    def get_tril_positions(n):
        """Lower square triangular position"""
        tril_pos = np.tril((np.arange(n ** 2) + 1).reshape(n, -1)).T.ravel()
        return tril_pos[tril_pos != 0]

    def plot_3d(self, fit_par_names):
        # Get data
        brute_results = self.unpickle_brute_results()
        hoppin_results = self.unpickle_hoppin_results()
        hoppin_paths = hoppin_results['paths']
        hoppin_minima = hoppin_results['minima']
        hoppin_final_result = hoppin_results['final_result']

        # Get plot positions
        n = len(fit_par_names) - 1
        combos = list(combinations(fit_par_names, 2))
        positions = self.get_tril_positions(n)

        for (xname, yname), pos in zip(combos, positions):
            # Specify subplot
            ax = plt.subplot(n, n, pos)

            # Find index for the two paratemers
            xi = fit_par_names.index(xname)
            yi = fit_par_names.index(yname)

            # get the meshgrids for x and y
            X = brute_results[2][xi]
            Y = brute_results[2][yi]

            # Find other axes to collapse
            axes = tuple([ii for ii in range(brute_results[3].ndim) if ii not in (xi, yi)])

            # Collapse to minimum Jout
            min_jout = np.amin(brute_results[3], axis=axes)
            min_xgrid = np.amin(X, axis=axes)
            min_ygrid = np.amin(Y, axis=axes)

            # Create heatmap
            ax.pcolormesh(min_xgrid, min_ygrid, min_jout)

            # Add basin hopping paths, minima, and final result
            ax.scatter(hoppin_paths[xname], hoppin_paths[yname],
                       marker='.', s=10, color='yellow')
            ax.scatter(hoppin_minima[xname], hoppin_minima[yname],
                       marker='*', s=10, color='red')
            ax.scatter(hoppin_final_result[yname], hoppin_final_result[xname],
                       marker='*', s=50, color='blue')

            # Add labels to edge only
            if pos >= n ** 2 - n:
                plt.xlabel(xname)
            if pos % n == 1:
                plt.ylabel(yname)

        plt.tight_layout()
        plt.savefig(self.file_path + 'heatmap.png')
