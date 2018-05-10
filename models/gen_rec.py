import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class GenRec(object):
    def __init__(self, parameters, full_genrec_path):
        self.save_path = full_genrec_path
        self.parameters = parameters
        try:  # read in genrec if one exists already
            self.genrec = pd.read_csv(self.save_path, index_col=0)
        except:  # otherwise, create a new one
            self.genrec = pd.DataFrame(columns=['sID', 'learning_style', 'mix_probs', 'fit_par',
                                                'NLL', 'BIC', 'AIC'] +
                                               [par + '_gen' for par in parameters.par_names] +
                                               [par + '_rec' for par in parameters.par_names])
        self.genrec_row = len(self.genrec)

    def update_and_save_genrec(self, gen_pars, rec_pars, fit, agent_stuff):
        row = np.concatenate(([agent_stuff['id'],
                               agent_stuff['learning_style'],
                               agent_stuff['mix_probs'], agent_stuff['fit_par']],
                               fit, gen_pars, rec_pars))
        self.genrec.loc[self.genrec_row, :] = row
        self.genrec_row += 1
        self.genrec.to_csv(self.save_path)

    def create_simulated_agent(self, agent_id, task_stuff, agent_stuff, save_agent_path):
        from fit_parameters import FitParameters
        import os

        agent_stuff['id'] = agent_id
        fit_params = FitParameters(parameters=self.parameters,
                                   task_stuff=task_stuff,
                                   agent_stuff=agent_stuff)

        # Simulate data with random parameter values (save Qs and action probs in agent_data)
        gen_pars = self.parameters.create_random_params(scale='lim', get_all=True, mode='soft')
        agent_data = fit_params.get_agent_data(way='simulate',
                                               all_params_lim=gen_pars)

        # Write the created file
        if not os.path.isdir(save_agent_path):
            os.makedirs(save_agent_path)
        fit_params.write_agent_data(agent_data=agent_data,
                                    save_path=save_agent_path,
                                    file_name='AL_' + agent_stuff['learning_style'])

        return agent_data
