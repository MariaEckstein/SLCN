import numpy as np


class BayesAgent(object):
    def __init__(self, agent_stuff, task_stuff):
        self.id = agent_stuff['id']
        self.prior = agent_stuff['prior']
        self.reward_prob = task_stuff['reward_prob']
        self.p_switch = self.prior.copy()
        self.selected_box = np.random.choice(range(2))
        
    def take_action(self):
        return self.selected_box

    def learn(self, action, reward):
        if reward == 1:
            self.p_switch = self.prior.copy()
        else:
            lik_switch = 1 * self.p_switch
            lik_noswitch = (1 - self.reward_prob) * (1 - self.p_switch)
            self.p_switch = lik_switch / (lik_switch + lik_noswitch)

        print('p_switch', self.p_switch)
        if self.p_switch > 0.5:  # np.random.rand():
            print('switch!')
            self.selected_box = 1 - self.selected_box
            self.p_switch = self.prior.copy()
