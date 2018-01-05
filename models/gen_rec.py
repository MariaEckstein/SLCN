import pandas as pd
import numpy as np


class GenRec(object):
    def __init__(self, parameters, full_genrec_path):
        self.save_path = full_genrec_path
        try:  # read in genrec if one exists already
            self.genrec = pd.read_csv(self.save_path, index_col=0)
        except:  # otherwise, create a new one
            self.genrec = pd.DataFrame(columns=['sID',
                                                'learning_style', 'method', 'fit_par',
                                                'NLL', 'BIC', 'AIC'] +
                                               [par + '_gen' for par in parameters.par_names] +
                                               [par + '_rec' for par in parameters.par_names])
        self.genrec_row = len(self.genrec)

    def update_and_save_genrec(self, gen_pars, rec_pars, fit, agent_stuff):
        print('gen_pars:', np.round(gen_pars, 2), '\nrec_pars:', np.round(rec_pars, 2), '\nfit:', fit)
        row = np.concatenate(([agent_stuff['id'],
                               agent_stuff['learning_style'], agent_stuff['method'], agent_stuff['fit_par']],
                              fit, gen_pars, rec_pars))
        self.genrec.loc[self.genrec_row, :] = row
        self.genrec_row += 1
        self.genrec.to_csv(self.save_path)
