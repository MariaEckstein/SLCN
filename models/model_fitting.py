import numpy as np
import pandas as pd
from universal_agent import UniversalAgent
from task import Task
from history import History
from scipy.optimize import minimize, basinhopping


class ModelFitting(object):

    def __init__(self, agent_stuff, task_stuff):
        self.agent_stuff = agent_stuff
        self.task_stuff = task_stuff
        pars = ['alpha', 'beta', 'epsilon', 'perseverance', 'decay']
        self.genrec = pd.DataFrame(columns=['sID', 'learning_style', 'method', 'NLL', 'BIC', 'AIC'] +
                                           [par + '_gen' for par in pars] + [par + '_rec' for par in pars])
        self.genrec_row = 0

    def update_genrec_and_save(self, row, file_path):
        self.genrec.loc[self.genrec_row, :] = row
        self.genrec_row += 1
        self.genrec.to_csv(file_path)

    def simulate_agents(self, ag, params):

        goal = 'simulate'

        task = Task(self.task_stuff, params, goal, ag, 200)
        agent = UniversalAgent(self.agent_stuff, params, task, ag)
        hist = History(task, agent)

        for trial in range(1, task.n_trials):
            task.switch_box(trial, goal)
            action = agent.take_action(trial, goal)
            reward = task.produce_reward(action, trial, goal)
            agent.learn(action, reward)
            hist.update(agent, task, action, reward, trial)

        hist.save_csv()

        return np.array([agent.alpha, agent.beta, agent.epsilon, agent.perseverance, agent.decay])

    def calculate_fit(self, params, ag, only_NLL):

        goal = 'model'

        task = Task(self.task_stuff, self.agent_stuff, goal, ag, 200)
        agent = UniversalAgent(self.agent_stuff, params, task, ag)

        for trial in range(1, task.n_trials):
            task.switch_box(trial, goal)
            action = agent.take_action(trial, goal)
            reward = task.produce_reward(action, trial, goal)
            agent.learn(action, reward)

        n_fit_params = sum(self.agent_stuff['free_par'])

        BIC = - 2 * agent.LL + n_fit_params * np.log(task.n_trials)
        AIC = - 2 * agent.LL + n_fit_params

        if only_NLL:
            return -agent.LL
        else:
            return [-agent.LL, BIC, AIC]

    def get_fit_par(self, best_par):
        fit_par = self.agent_stuff['default_par']
        j = 0
        for i, par in enumerate(fit_par):
            if self.agent_stuff['free_par'][i]:
                fit_par[i] = best_par[j]
                j += 1
        return np.array(fit_par)

    def minimize_NLL(self, ag, n_fit_par, n_iter):
        values = np.zeros([n_iter, n_fit_par + 1])
        for iter in range(n_iter):
            params0 = np.random.rand(n_fit_par)
            minimization = minimize(self.calculate_fit,
                                    params0, (ag, True),  # arguments calculate_fit: params, id, only_NLL
                                    method='Nelder-Mead')
            values[iter, :] = np.concatenate(([minimization.fun], minimization.x))
        minimum = values[:, 0] == min(values[:, 0])
        best_par = values[minimum][0]
        return best_par[1:]

    # class MyBounds(object):
    #     def __init__(self, xmax=(1, 1, 1, 1, 1, 1), xmin=(0, 0, 0, 0, 0, 0)):
    #         self.xmax = np.array(xmax)
    #         self.xmin = np.array(xmin)
    #
    #     def __call__(self, **kwargs):
    #         x = kwargs["x_new"]
    #         tmax = bool(np.all(x <= self.xmax))
    #         tmin = bool(np.all(x >= self.xmin))
    #         return tmax and tmin
    #
    # my_bounds = MyBounds()


# class RandomDisplacementBounds(object):
#     """random displacement with bounds"""
#
#     def __init__(self, params0, xmax=np.ones(n_fit_par), xmin=np.zeros(n_fit_par), stepsize=0.2):
#         self.xmin = xmin
#         self.xmax = xmax
#         self.stepsize = stepsize
#         self.params0 = params0
#
#     def __call__(self, x):
#         """take a random step but ensure the new position is within the bounds"""
#         print(x)
#         for guess in range(100):
#             xnew = x + np.random.uniform(-self.stepsize, self.stepsize, np.shape(x))
#             if np.all(xnew < self.xmax) and np.all(xnew > self.xmin):
#                 return xnew
#             else:
#                 return self.params0
#
#
# # define the new step taking routine and pass it to basinhopping
# take_step = RandomDisplacementBounds(params0)
# minimization = basinhopping(model.calculate_fit, params0, niter=20, T=0.4, stepsize=0.2,
#                             minimizer_kwargs={'method': 'Nelder-Mead', 'args': (ag, '', True)},
#                             take_step=take_step)
# print(sim_par)
# print(minimization.x)

# minimization = basinhopping(model.calculate_fit, params0, niter=10,  # niter : The number of basin hopping iterations
#                    T=0.4, stepsize=0.4, niter_success=5,  # T : The “temperature” parameter for the accept or reject criterion. Higher “temperatures” mean that larger jumps in function value will be accepted. For best results T should be comparable to the separation (in function value) between local minima.
#                                                                                 # stepsize : initial step size for use in the random displacement.
#                    minimizer_kwargs={'method': 'Nelder-Mead', 'args': (ag, '', True)})  # , accept_test=my_bounds
