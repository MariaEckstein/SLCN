
class ModelFitting(object):

    def calculate_NLL(self, params, agent_id, agent_name):

        import numpy as np
        from universal_agent import UniversalAgent
        from task import Task

        task_stuff = {'n_actions': 2,
                      'p_reward': 0.75,
                      'path': 'C:/Users/maria/MEGAsync/SLCN/ProbabilisticSwitching/Prerandomized sequences'}
        agent_stuff = {'id': agent_id,
                       'name': agent_name,
                       'data_path': 'C:/Users/maria/MEGAsync/SLCNdata/PSResults',
                       'alpha': params[0],
                       'beta': params[1],
                       'epsilon': params[2],
                       'perseverance': params[3],
                       'decay': params[4],
                       'method': 'softmax'}
        task = Task(task_stuff, agent_stuff, 'model_data', agent_id, 150)
        agent = UniversalAgent(agent_stuff, task, 'model_data')

        for trial in range(1, task.n_trials):
            task.switch_box(trial)
            action = agent.take_action(trial)
            reward = task.produce_reward(action, trial)
            agent.learn(action, reward)

        n_params = sum([param == 0.5 for param in params])
        BIC = -agent.LL + n_params * np.log(task.n_trials)
        AIC = -agent.LL + n_params

        return -agent.LL
