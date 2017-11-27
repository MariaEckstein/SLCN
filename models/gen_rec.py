import numpy as np
from model_fitting import ModelFitting


class GenRec(object):
    def __init__(self, parameters):
        self.parameters = parameters

    def generate_and_recover(self, fit_pars, agents, n_iter, agent_stuff, task_stuff, use='fit_human_data'):
        agent_stuff = self.__update_agent_stuff(agent_stuff, fit_pars, use='fit_human_data')
        n_fit_par = sum(agent_stuff['free_par'])
        gen_par = np.full(n_fit_par, np.nan)
        model = ModelFitting(agent_stuff, task_stuff, self.parameters)
        model.adjust_free_par()
        for ag in agents:
            # Generate
            if use == 'generate_and_recover':
                gen_par = self.parameters.inverse_sigmoid(np.random.rand(n_fit_par))  # get random values [-inf to inf]
                model.simulate_agent(gen_par, ag)
            # Recover
            rec_par = model.minimize_NLL(ag, n_iter)
            fit = model.simulate_agent(rec_par, ag, 'calculate_fit')
            # Print and save
            model.update_genrec(gen_par, rec_par, fit, ag)
            print('Fitted pars:', fit_pars,
                  'Model:', agent_stuff['learning_style'], agent_stuff['method'], ', ',
                  'Agent:', ag)

    def vary_parameters(self, param_name, agent_stuff, task_stuff):
        agent_stuff['data_path'] = 'C:/Users/maria/MEGAsync/SLCNdata/' + '_vary_' + param_name
        model = ModelFitting(agent_stuff, task_stuff, self.parameters)
        for ag, par in enumerate(np.arange(0.01, 0.99, 0.15)):
            gen_par = self.parameters.inverse_sigmoid(par)
            model.simulate_agent([gen_par], ag)

    def __update_agent_stuff(self, agent_stuff, fit_pars, use='fit_human_data'):
        agent_stuff['free_par'] = [np.any(par == np.array(fit_pars)) for par in self.parameters.par_names]
        if use == 'generate_and_recover':
            data_path_extension = agent_stuff['learning_style'] + '/' + agent_stuff['method'] + '_fit_' + ''.join(fit_pars)
            hist_path_extension = ''
        else:
            data_path_extension = 'PSResults'
            hist_path_extension = '/' + agent_stuff['learning_style'] + '/' + agent_stuff['method']
        agent_stuff['data_path'] = 'C:/Users/maria/MEGAsync/SLCNdata/' + data_path_extension
        agent_stuff['hist_path'] = agent_stuff['data_path'] + hist_path_extension
        return agent_stuff
