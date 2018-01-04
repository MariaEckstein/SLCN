import numpy as np


class Parameters(object):
    def __init__(self, par_names, fit_pars, par_hard_limits, par_soft_limits, default_pars_lim):
        self.par_names = par_names
        self.fit_pars = fit_pars
        self.par_hard_limits = par_hard_limits
        self.par_soft_limits = par_soft_limits
        self.default_pars_lim = default_pars_lim

    def inf_to_lim(self, pars_inf):
        all_pars_inf = self.get_all_pars(pars_inf)
        all_pars_01 = self.sigmoid(all_pars_inf)
        all_pars_lim = self.constrain_limits(all_pars_01)
        return all_pars_lim

    def lim_to_inf(self, all_pars_lim):
        all_pars_01 = self.remove_limits(all_pars_lim)
        all_pars_inf = self.inverse_sigmoid(all_pars_01)
        pars_inf = [par for i, par in enumerate(all_pars_inf) if self.fit_pars[i]]
        return pars_inf

    def create_random_params(self, scale='lim', get_all=True, mode='hard'):
        pars_01 = np.random.rand(sum(self.fit_pars))
        if get_all:
            pars_inf = self.inverse_sigmoid(pars_01)
            all_pars_inf = self.get_all_pars(pars_inf)
            pars_01 = self.sigmoid(all_pars_inf)
        if scale == 'lim':
            if mode == 'soft':
                return self.constrain_limits(pars_01, mode='soft')
            else:
                return self.constrain_limits(pars_01)
        elif scale == 'inf':
            return self.inverse_sigmoid(pars_01)

    def adjust_fit_pars(self, learning_style, method):
        if learning_style == 'Bayes':
            self.fit_pars[0] = False  # alpha
        if method == 'epsilon-greedy':
            self.fit_pars[1:3] = [False, True]  # beta, epsilon
        elif method == 'softmax':
            self.fit_pars[1:3] = [True, False]  # beta, epsilon
        elif method == 'direct':
            self.fit_pars[1:3] = [False, False]  # beta, epsilon

    def get_all_pars(self, params_inf):
        default_pars_01 = self.remove_limits(self.default_pars_lim)
        all_pars_inf = self.inverse_sigmoid(default_pars_01)
        j = 0
        for i, par in enumerate(all_pars_inf):
            if self.fit_pars[i]:
                all_pars_inf[i] = params_inf[j]
                j += 1
        return np.array(all_pars_inf)

    @staticmethod
    def inverse_sigmoid(pars_01):
        pars_inf = -np.log(1 / pars_01 - 1)
        return pars_inf

    @staticmethod
    def sigmoid(pars_inf):
        pars_01 = 1 / (1 + np.e ** -pars_inf)
        return pars_01

    def constrain_limits(self, pars_01, mode='hard'):
        if mode == 'soft':
            limits = self.par_soft_limits
        else:
            limits = self.par_hard_limits
        pars_lim = pars_01.copy()
        for i in range(len(pars_lim)):
            width = limits[i][1] - limits[i][0]
            pars_lim[i] = width * pars_lim[i] + limits[i][0]
        return pars_lim

    def remove_limits(self, pars_lim, mode='hard'):
        if mode == 'soft':
            limits = self.par_soft_limits
        else:
            limits = self.par_hard_limits
        pars_01 = pars_lim.copy()
        for i in range(len(pars_01)):
            width = limits[i][1] - limits[i][0]
            pars_01[i] = (pars_01[i] - limits[i][0]) / width
        return pars_01
