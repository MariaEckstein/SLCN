import numpy as np


class TransformPars(object):

    @staticmethod
    def get_pars(agent_stuff, params):
        pars = agent_stuff['default_par'].copy()
        j = 0
        for i, par in enumerate(pars):
            if agent_stuff['free_par'][i]:
                pars[i] = params[j]
                j += 1
        return np.array(pars)

    @staticmethod
    def adjust_limits(pars):
        pars = [10 * par if i == 1 or i == 3 else par for i, par in enumerate(pars)]  # beta & persev. are 0 to 20
        pars[2] = 0.5 * pars[2]  # large epsilon makes it impossible to recover anything
        return pars

    @staticmethod
    def inverse_sigmoid(pars):
        return -np.log(1 / pars - 1)  # transform [0, 1] into [-inf, inf]

    @staticmethod
    def sigmoid(pars):
        return 1 / (1 + np.e ** -pars)  # transform [-inf, inf] into [0, 1]
