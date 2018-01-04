import pandas as pd
import numpy as np
import os


class GenRec(object):
    def __init__(self, parameters, save_path):
        self.genrec = pd.DataFrame(columns=['sID', 'learning_style', 'method', 'NLL', 'BIC', 'AIC'] +
                                           [par + '_gen' for par in parameters.par_names] +
                                           [par + '_rec' for par in parameters.par_names])
        self.genrec_row = 0
        self.save_path = save_path
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

    def update_and_save_genrec(self, gen_pars, rec_pars, fit, agent_stuff):
        print('gen_pars:', np.round(gen_pars, 2), '\nrec_pars:', np.round(rec_pars, 2), '\nfit:', fit)
        row = np.concatenate(([agent_stuff['id'], agent_stuff['learning_style'], agent_stuff['method']],
                              fit, gen_pars, rec_pars))
        self.genrec.loc[self.genrec_row, :] = row
        self.genrec_row += 1
        self.genrec.to_csv(self.save_path + '/genrec.csv')
