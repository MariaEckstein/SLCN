import numpy as np


class SimulateInteractive(object):
    def __init__(self, agent):
        self.agent = agent
        self.task_context = np.nan
        self.task_alien = np.nan
        self.suggested_action = np.nan
        self.TS_values = np.nan
        self.action_values = np.nan

    def trial(self, trial):
        print('\n\tTRIAL {0}'.format(str(trial)))
        self.task_context = int(input('Context (0, 1, 2):'))
        self.task_alien = int(input('Alien (0, 1, 2, 3):'))
        stimulus = [self.task_context, self.task_alien]
        self.suggested_action = self.agent.select_action(stimulus)  # calculate p_actions
        return stimulus

    def print_values_pre(self):
        self.TS_values = np.round(self.agent.Q_high[self.task_context, :], 2)
        self.action_values = np.round(self.agent.Q_low[:, self.task_alien, :], 2)
        print('TS.     all values:   {0}; all ps:   {1}\n'
              'Action. comb. values: {3}; comb. ps: {4}; suggested: {2}\n'
              '\tall values:\n{5}'.format(
                str(self.TS_values), str(np.round(self.agent.p_TS, 2)),
                str(self.suggested_action),
                str(np.round(self.agent.Q_actions, 2)), str(np.round(self.agent.p_actions, 2)),
                str(self.action_values)))

    def print_values_post(self, action, reward, correct):
        TS_values_new = np.round(self.agent.Q_high[self.task_context, :], 2)
        action_values_new = np.round(self.agent.Q_low[:, self.task_alien, :], 2)
        print('Reward: {0} ({1})\n'
              'Q_TS.     Old: {2}, RPE: {3}, p_TS: {4}, New: {5}\n'
              'Q_action. Old: {6}, RPE: {7}, p_ac: {8}, New: {9}'.format(
                str(np.round(reward, 2)), str(correct),
                str(self.TS_values), str(np.round(self.agent.RPEs_high, 2)),
                str(np.round(self.agent.p_TS, 2)), str(TS_values_new),
                str(self.action_values[:, action]), str(np.round(self.agent.RPEs_low, 2)),
                str(np.round(self.agent.p_TS, 2)), str(action_values_new[:, action])))
