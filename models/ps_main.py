import numpy as np
from model_fitting import ModelFitting
from transform_pars import TransformPars


# Set parameters
n_agents = 50
n_iter = 10

# Info about task and agent
task_stuff = {'n_actions': 2,
              'p_reward': 0.75,
              'path': 'C:/Users/maria/MEGAsync/SLCN/ProbabilisticSwitching/Prerandomized sequences'}
pars = ['alpha', 'beta', 'epsilon', 'perseverance', 'decay']
n_par = 5
trans = TransformPars()

# Check that parameters are working
agent_stuff = {'default_par': [-1, -3, -100, -100, -100],  # after transform: [0.27, 0.47, 0, 0, 0]
               'n_agents': 1}  # number of simulated agents
def check_paramters(param_name, agent_stuff, task_stuff):
    if param_name == 'perseverance':
        default_epsilon = 0
    else:
        default_epsilon = -100
    agent_stuff['default_par'][2] = default_epsilon
    trans = TransformPars()
    agent_stuff['free_par'] = [par == param_name for par in pars]
    for learning_style in ['RL', 'Bayes']:
        agent_stuff['learning_style'] = learning_style
        for method in ['direct', 'epsilon-greedy', 'softmax']:
            agent_stuff['method'] = method
            model = ModelFitting(agent_stuff, task_stuff, '_vary_' + param_name)
            for ag, alpha in enumerate(np.arange(0.01, 0.99, 0.15)):
                print(alpha)
                alpha_gen = trans.inverse_sigmoid(alpha)
                model.simulate_agents([alpha_gen], 4 * ag)  # 4 * ag to get the same randomization for everyone

for param_name in pars:
    print(param_name)
    # check_paramters(param_name, agent_stuff, task_stuff)

# Generate and recover
agent_stuff['free_par'] = [True, False, False, False, False]  # number of simulated agents
for learning_style in ['RL', 'Bayes']:
    agent_stuff['learning_style'] = learning_style
    for method in ['direct', 'softmax', 'epsilon-greedy']:
        agent_stuff['method'] = method
        model = ModelFitting(agent_stuff, task_stuff, '_fit_alpha')
        model.adjust_free_par()

        for ag in range(n_agents):
            n_fit_par = sum(agent_stuff['free_par'])
            print('Learning:', learning_style, '  Method:', method, '  Agent:', ag, '  n_fit_par:', n_fit_par)
            # Generate
            rand_par_gen = trans.inverse_sigmoid(np.random.rand(n_fit_par))  # make 0 to 1 into -inf to inf
            model.simulate_agents(rand_par_gen, ag)
            # Recover
            best_par = model.minimize_NLL(ag, n_fit_par, n_iter)
            # Print and save
            gen_par = trans.get_pars(agent_stuff, rand_par_gen)
            rec_par = trans.get_pars(agent_stuff, best_par)
            gen_par_01 = trans.adjust_limits(trans.sigmoid(gen_par))
            rec_par_01 = trans.adjust_limits(trans.sigmoid(rec_par))
            print('gen_par:', gen_par_01, '\nrec_par:', rec_par_01)
            NLL, BIC, AIC = model.simulate_agents(rec_par, ag, 'calculate_fit')
            model.update_genrec(
                np.concatenate(([ag, learning_style, method, NLL, BIC, AIC], gen_par_01, rec_par_01)),
                agent_stuff['data_path'] + '/genrec.csv')

# Fit parameters to actual data
# agents = np.array([15, 16, 17, 18, 20, 22, 23, 24, 25, 31, 32, 33, 34, 35, 40, 42, 43, 44, 48, 49, 52, 53, 56, 60, 63, 65, 67])
#
# params0 = 0.5 * np.ones(len(agent_stuff['free_par']))
# NLL, BIC, AIC = model.calculate_fit(params0, agents[0], False)
# minimization = minimize(model.calculate_fit,
#                         params0, (agents[0], True),  # arguments to model.calculate_fit: id, only_NLL
#                         method='Nelder-Mead', options={'disp': True})
# fit_par = model.get_fit_par(minimization)
# NLL, BIC, AIC = model.calculate_fit(fit_par, agents[0], False)
# model.simulate_agents(fit_par)
