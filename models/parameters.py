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
        all_pars_01 = self.change_scale(all_pars_inf, 'inf_to_01')
        all_pars_lim = self.change_limits(all_pars_01, '01_to_lim')
        return all_pars_lim

    def lim_to_inf(self, all_pars_lim):
        all_pars_01 = self.change_limits(all_pars_lim, 'lim_to_01')
        all_pars_inf = self.change_scale(all_pars_01, '01_to_inf')
        pars_inf = [par for i, par in enumerate(all_pars_inf) if self.fit_pars[i]]
        return pars_inf

    def create_random_params(self, scale='lim', get_all=True, mode='hard'):
        if get_all and (scale == 'lim'):
            if mode == 'soft':
                limits = self.par_soft_limits
            elif mode == 'hard':
                limits = self.par_hard_limits
            all_pars_lim = [limit[0] + np.random.rand() * (limit[1] - limit[0]) for limit in limits]
            return np.array(all_pars_lim)
        elif (not get_all) and (scale == 'inf'):
            pars_01 = np.random.rand(sum(self.fit_pars))
            return np.array(self.change_scale(pars_01, '01_to_inf'))

    @staticmethod
    def change_scale(pars_in, direction):
        pars_out = pars_in.copy()
        if direction == '01_to_inf':
            assert(np.all(pars_out >= 0) and np.all(pars_out <= 1))
            pars_out = 0.99999 * pars_out + 0.00001 * 0.5 * np.ones(len(pars_out))  # Avoid 0's and 1's
            pars_out = -np.log(1 / pars_out - 1)  # inverse sigmoid
        elif direction == 'inf_to_01':
            pars_out = 0.99999 * pars_out + 0.00001 * 0.5 * np.ones(len(pars_out))  # Avoid 0's and 1's
            pars_out = 1 / (1 + np.e ** -pars_out)  # sigmoid
            assert (np.all(pars_out >= 0) and np.all(pars_out <= 1))
        return pars_out

    def change_limits(self, all_pars_in, direction, mode='hard'):
        if mode == 'soft':
            limits = self.par_soft_limits
        else:
            limits = self.par_hard_limits

        all_pars_out = all_pars_in.copy()
        for i in range(len(all_pars_out)):
            width = limits[i][1] - limits[i][0]
            if direction == '01_to_lim':
                all_pars_out[i] = width * all_pars_out[i] + limits[i][0]
            elif direction == 'lim_to_01':
                all_pars_out[i] = (all_pars_out[i] - limits[i][0]) / width
        return all_pars_out

    def set_fit_pars(self, fit_pars):
        self.fit_pars = fit_pars

    def adjust_fit_pars(self, method, learning_style=np.nan):
        # if learning_style == 'RL':
        #     self.fit_pars[np.array(self.par_names) == 'alpha'] = True
        if learning_style == 'Bayes':
            self.fit_pars[np.array(self.par_names) == 'alpha'] = False
        # elif learning_style == 'flat':
        #     self.fit_pars[np.array(self.par_names) == 'mix'] = False  # mix RPE_low and RPE_high
        if method == 'epsilon-greedy':
            self.fit_pars[np.array(self.par_names) == 'beta'] = False
            # self.fit_pars[np.array(self.par_names) == 'epsilon'] = True
        elif method == 'softmax':
            # self.fit_pars[np.array(self.par_names) == 'beta'] = True
            self.fit_pars[np.array(self.par_names) == 'epsilon'] = False

    def get_all_pars(self, params_inf):
        default_pars_01 = self.change_limits(self.default_pars_lim, 'lim_to_01')
        all_pars_inf = self.change_scale(default_pars_01, '01_to_inf')
        j = 0
        for i, par in enumerate(all_pars_inf):
            if self.fit_pars[i]:
                all_pars_inf[i] = params_inf[j]
                j += 1
        return np.array(all_pars_inf)
