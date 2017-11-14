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
    def inverse_sigmoid(pars):
        """
        Produce parameters that go INTO the simulation -> [-inf, inf] because that is what Nelder-Mead likes
        """
        return -np.log(1 / pars - 1)  # transform [0, 1] into [-inf, inf]

    @staticmethod
    def sigmoid(pars):
        """"
        Apply to parameters WITHIN universal_agent -> [0, 1] because fixed ranges are what the agent likes
        """
        return 1 / (1 + np.e ** -pars)  # transform [-inf, inf] into [0, 1]

    @staticmethod
    def adjust_limits(pars):
        """
        Get the specific ranges of each parameter
        """
        pars[1] = 1 + pars[1] * 5  # beta [1, 6]
        pars[2] = pars[2] / 4  # epsilon [0, 0.15]
        pars[3] = 0.6 * pars[3] - 0.3  # perseverance [-0.3, 0.3]
        pars[4] = 0.3 * pars[4]  # decay [0, 0.3]
        return pars
