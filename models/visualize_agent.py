import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from fit_parameters import FitParameters


class VisualizeAgent:
    def __init__(self, parameters, agent_name):
        self.parameters = parameters
        self.agent_name = agent_name

    def plot_Qs(self, par_name, parameter_values, fit_params, p_or_Q):
        # Get default parameter values
        gen_pars = self.parameters.default_pars_lim
        par_index = np.argwhere(np.array(self.parameters.par_names) == par_name)
        for par_value in parameter_values:
            # Replace the parameter in question with the value of interest and simulate data
            gen_pars[par_index] = par_value
            print("Simulating data...")
            agent_data = fit_params.get_agent_data(way='simulate',
                                                   all_params_lim=gen_pars)
            # Plot Q values over time
            for TS_or_action in ['_TS', '_action']:
                columns = [col for col in agent_data.columns if p_or_Q + TS_or_action in col]

                print("Melting data...")
                agent_data_long = pd.melt(agent_data,
                                          id_vars=["trial_index", "reward", "item_chosen", "sad_alien", "TS", "context"],
                                          value_vars=columns,
                                          var_name=TS_or_action, value_name=p_or_Q)
                print("Plotting data...")
                TS_plot = sns.lmplot(x="trial_index", y=p_or_Q, hue=TS_or_action, col='context',
                           scatter=True, fit_reg=False,
                           size=5, data=agent_data_long)
                TS_plot.savefig(TS_or_action + "plot.png")

                ax = plt.gca()
                ax.set_title('Pars: {0}, {1}-values, {2}'.format(str(np.round(gen_pars, 2)), TS_or_action, p_or_Q))
                plt.show()

    def plot_generated_recovered_Qs(self, task_stuff, method, learning_style, mix_probs):
        # Specify model
        self.parameters.set_fit_pars(np.zeros(len(self.parameters.par_names), dtype=bool))  # fix all parameters; no fitting
        agent_stuff = {'name': 'alien',
                       'n_TS': 3,
                       'method': method,
                       'learning_style': learning_style,
                       'id': 0,
                       'fit_par': [],
                       'mix_probs': mix_probs}
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
        if self.agent_name == 'alien':
            sub = 1
            for sad_alien in range(3):
                for context in range(3):
                    plot_dat = agent_data.loc[(agent_data['sad_alien'] == sad_alien) & (agent_data['context'] == context)]
                    for col in ['Q_low_TS0_action_{0}'.format(str(i)) for i in range(3)]:
                        plt.subplot(9, 3, sub)
                        plt.plot(plot_dat[col], plot_dat[col + '_rec'], 'o')
                        plt.title('alien{0} context{1} {2}'.format(str(sad_alien), str(context), col))
                        sub += 1
            plt.show()
            sub = 1
            for TS in range(3):
                plot_dat = agent_data.loc[agent_data['TS'] == TS]
                for col in ['Q_low_TS0_action_{0}'.format(str(i)) for i in range(3)]:
                    plt.subplot(3, 3, sub)
                    plt.plot(plot_dat[col], plot_dat[col + '_rec'], 'o')
                    plt.title('TS{0} {1}'.format(str(TS), col))
                    sub += 1
            plt.show()

            for i, col in enumerate(['p_action' + str(i) for i in range(3)]):
                plt.subplot(1, 3, i + 1)
                plt.plot(agent_data[col], agent_data[col + '_rec'], 'o')
                plt.title(col)
            plt.show()

        else:
            for col in ['values_l', 'values_l', 'p_action_l']:
                plt.plot(agent_data[col], agent_data[col + '_rec'], 'o')
            plt.show()
