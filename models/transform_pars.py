import numpy as np


class TransformPars(object):

    @staticmethod
    def get_pars(agent_stuff, params):
        pars = np.array(agent_stuff['default_par'])  # 0's and 1's like before - no transformation
        j = 0
        for i, par in enumerate(pars):
            if agent_stuff['free_par'][i]:
                pars[i] = params[j]
                j += 1
        return pars

    @staticmethod
    def transform_pars(pars):
        pars = 1 / (1 + np.e ** -pars)  # sigmoid -> make -inf to inf into 0 to 1
        pars = [20 * par if i == 1 or i == 3 else par for i, par in enumerate(pars)]  # beta & persev. are 0 to 20
        return pars
