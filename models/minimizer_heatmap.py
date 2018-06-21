import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from itertools import combinations


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
    def __init__(self, heatmap_data, hoppin_minima, hoppin_paths, final_result):
        self.heatmap_data = heatmap_data
        self.hoppin_minima = hoppin_minima
        self.hoppin_paths = hoppin_paths
        self.final_result = final_result

    def plot(self, path):

        # Prepare figure
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')

        # Plot heatmap
        sns.heatmap(self.heatmap_data)

        # Add paths
        x = self.hoppin_paths['beta'] * self.heatmap_data.shape[0]
        y = self.hoppin_paths['alpha'] * self.heatmap_data.shape[0]
        ax.scatter(x, y, marker='.', s=10, color='yellow')

        # Add minima
        x = self.hoppin_minima['beta'] * self.heatmap_data.shape[0]
        y = self.hoppin_minima['alpha'] * self.heatmap_data.shape[0]
        ax.scatter(x, y, marker='*', s=10, color='red')

        # Add final result
        x = self.final_result['alpha']
        y = self.final_result['beta']
        ax.scatter(x, y, marker='*', s=50, color='blue')

        # Save figure as png
        plt.savefig(path + 'heatmap.png')

    @staticmethod
    def get_tril_positions(n):
        """Lower square triangular position"""
        tril_pos = np.tril((np.arange(n ** 2) + 1).reshape(n, -1)).T.ravel()
        return tril_pos[tril_pos != 0]

    def plot_3d(self, brute_results, fit_par_names, path=None):
        n = len(fit_par_names) - 1
        combos = list(combinations(fit_par_names, 2))
        positions = self.get_tril_positions(n)

        for (xname, yname), pos in zip(combos, positions):
            # Specify subplot
            ax = plt.subplot(n, n, pos)

            # Find index for these variables
            xi = fit_par_names.index(xname)
            yi = fit_par_names.index(yname)

            # get the meshgrids for x and y
            X = brute_results[2][xi]
            Y = brute_results[2][yi]

            # Find other axis to collapse
            axes = tuple([ii for ii in range(brute_results[3].ndim) if ii not in (xi, yi)])

            # Collapse to minimum Jout
            min_jout = np.amin(brute_results[3], axis=axes)
            min_xgrid = np.amin(X, axis=axes)
            min_ygrid = np.amin(Y, axis=axes)

            ax.pcolormesh(min_xgrid, min_ygrid, min_jout)

            # Add colorbar to each plot
            # plt.colorbar()

            # Add labels to edge only
            if pos >= n ** 2 - n:
                plt.xlabel(xname)
            if pos % n == 1:
                plt.ylabel(yname)

        plt.tight_layout()
        # plt.show()
        plt.savefig(path + 'heatmap.png')