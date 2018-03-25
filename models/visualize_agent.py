import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from fit_parameters import FitParameters


class VisualizeAgent:
    def __init__(self, parameters):
        self.parameters = parameters

    def plot_Qs(self, par_name, parameter_values, fit_params):
        # Get default parameter values
        gen_pars = self.parameters.default_pars_lim
        par_index = np.argwhere(np.array(self.parameters.par_names) == par_name)
        for par_value in parameter_values:
            # Replace the parameter in question with the value of interest and simulate data
            gen_pars[par_index] = par_value
            agent_data = fit_params.get_agent_data(way='simulate',
                                                   all_params_lim=gen_pars)
            # Plot Q values over time
            for lh in ['high', 'low']:
                agent_data_long = pd.melt(agent_data,
                                          id_vars=["trial_index", "reward", "item_chosen", "sad_alien", "TS"],
                                          value_vars=["Q_" + lh + "0", "Q_" + lh + "1", "Q_" + lh + "2"],
                                          var_name="action/context", value_name="Q_" + lh)

                sns.lmplot(x="trial_index", y="Q_" + lh, hue="sad_alien",
                           col="TS", row="action/context",
                           scatter=True, fit_reg=False,
                           size=5, data=agent_data_long)
                ax = plt.gca()
                ax.set_title('Parameters: ' + str(np.round(gen_pars, 2)))
                plt.show()

    def plot_generated_recovered_Qs(self, task_stuff, method, learning_style):
        # Specify model
        self.parameters.set_fit_pars(np.zeros(len(self.parameters.par_names), dtype=bool))  # fix all parameters; no fitting
        agent_stuff = {'n_TS': 3,
                       'method': method,
                       'learning_style': learning_style,
                       'id': 0,
                       'fit_par': []}
        fit_params = FitParameters(parameters=self.parameters,
                                   task_stuff=task_stuff,
                                   agent_stuff=agent_stuff)
        # Simulate data with random parameter values (save Qs and action probs in agent_data)
        gen_pars = self.parameters.create_random_params(scale='lim', get_all=True, mode='soft')
        agent_data = fit_params.get_agent_data(way='simulate',
                                               all_params_lim=gen_pars)
        # Calculate Qs and action probs based on the given parameter values and add to agent_data
        agent_data = fit_params.calculate_NLL(params_inf=self.parameters.lim_to_inf(gen_pars),
                                              agent_data=agent_data,
                                              goal='add_decisions_and_fit')
        # Compare initial and recovered Qs and action probs
        for i, col in enumerate(['Q_low' + str(i) for i in range(3)] + ['Q_high' + str(i) for i in range(3)]):
            plt.subplot(2, 3, i + 1)
            plt.plot(agent_data[col], agent_data[col + '_rec'], 'o')
            plt.title(col)
        plt.show()

        for i, col in enumerate(['p_action' + str(i) for i in range(3)]):
            plt.subplot(1, 3, i + 1)
            plt.plot(agent_data[col], agent_data[col + '_rec'], 'o')
            plt.title(col)
        plt.show()