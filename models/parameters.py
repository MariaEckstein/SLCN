import numpy as np


class Parameters(object):
    def __init__(self, par_names, par_limits, default_pars):
        self.par_names = par_names
        self.par_limits = par_limits
        self.default_pars = self.inverse_sigmoid(self.remove_limits(default_pars))  # parameter scale to [-inf, inf] scale

    def get_pars(self, agent_stuff, params):
        pars = self.default_pars.copy()
        j = 0
        for i, par in enumerate(pars):
            if agent_stuff['free_par'][i]:
                pars[i] = params[j]
                j += 1
        return np.array(pars)

    @staticmethod
    def inverse_sigmoid(pars):
        """
        Produce parameters that go INTO the simulation -> get range [-inf, inf] because that is what Nelder-Mead likes
        """
        return -np.log(1 / pars - 1)  # transform [0, 1] into [-inf, inf]

    @staticmethod
    def sigmoid(pars):
        """"
        Apply to parameters WITHIN universal_agent -> get range [0, 1] because fixed ranges are what the agent likes
        """
        return 1 / (1 + np.e ** -pars)  # transform [-inf, inf] into [0, 1]

    def constrain_limits(self, pars):
        for i in range(len(pars)):
            width = self.par_limits[i][1] - self.par_limits[i][0]
            pars[i] = width * pars[i] + self.par_limits[i][0]
        return pars

    def remove_limits(self, pars):
        for i in range(len(pars)):
            width = self.par_limits[i][1] - self.par_limits[i][0]
            pars[i] = (pars[i] - self.par_limits[i][0]) / width
        return pars
