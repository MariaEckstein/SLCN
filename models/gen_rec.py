import numpy as np
from model_fitting import ModelFitting


class GenRec(object):
    def __init__(self, parameters):
        self.parameters = parameters

    def generate_and_recover(self, fit_pars, agents, n_iter, agent_stuff, task_stuff, use='fit_human_data'):
        if use == 'generate_and_recover':
            data_path_extension = agent_stuff['learning_style'] + '/' + agent_stuff['method'] + '_fit_' + ''.join(fit_pars)
        else:
            data_path_extension = 'PSResults'
        model = ModelFitting(agent_stuff, task_stuff, self.parameters, data_path_extension)
        model.adjust_free_par()
        for ag in agents:
            n_fit_par = sum(agent_stuff['free_par'])
            print('Fitted pars:', fit_pars, '(', n_fit_par, '), Model:', agent_stuff['learning_style'], agent_stuff['method'], ', Agent:', ag)
            # Generate
            gen_par = np.full(n_fit_par, np.nan)
            if use == 'generate_and_recover':
                gen_par = self.parameters.inverse_sigmoid(np.random.rand(n_fit_par))  # get random values [-inf to inf]
                model.simulate_agent(gen_par, ag)
            # Recover
            rec_par = model.minimize_NLL(ag, n_iter)
            fit = model.simulate_agent(rec_par, ag, 'calculate_fit')
            # Print and save
            model.update_genrec(gen_par, rec_par, fit, ag)

    def vary_parameters(self, param_name, agent_stuff, task_stuff):
        model = ModelFitting(agent_stuff, task_stuff, self.parameters, '_vary_' + param_name)
        for ag, par in enumerate(np.arange(0.01, 0.99, 0.15)):
            par_gen = self.parameters.inverse_sigmoid(par)
            model.simulate_agent([par_gen], ag)
