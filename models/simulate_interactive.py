import numpy as np


class SimulateInteractive(object):
    def __init__(self, agent, mix_probs):
        self.agent = agent
        self.task_context = np.nan
        self.task_alien = np.nan
        self.mix_probs = mix_probs
        self.TS_values = np.nan
        self.action_values = np.nan

    def print_values_pre(self, context, alien, suggested_action):
        self.task_context = context
        self.task_alien = alien
        self.TS_values = np.round(self.agent.Q_high[self.task_context, :], 2)
        self.action_values = np.round(self.agent.Q_low[:, self.task_alien, :], 2)
        if self.mix_probs:
            print('TS.     all values:   {0}; all ps:   {1}\n'
                  'Action. comb. values: {3}; comb. ps: {4}; suggested: {2}\n'
                  '\tall values:\n{5}'.format(
                    str(self.TS_values), str(np.round(self.agent.p_TS, 2)),
                    str(suggested_action),
                    str(np.round(self.agent.Q_actions, 2)), str(np.round(self.agent.p_actions, 2)),
                    str(self.action_values)))
        else:
            print('TS.     Selected:  {0}, value: {1} (all values: {2}; all ps: {3})\n'
                  'Action. Suggested: {4}, value: {5} (all values: {6}; all ps: {7})'.format(
                    str(self.agent.TS), str(self.TS_values[self.agent.TS]),
                    str(self.TS_values), str(np.round(self.agent.p_TS, 2)),
                    str(suggested_action), str(self.action_values[self.agent.TS, suggested_action]),
                    str(self.action_values[self.agent.TS, :]), str(np.round(self.agent.p_actions, 2))))

    def print_values_post(self, action, reward, correct):
        TS_values_new = np.round(self.agent.Q_high[self.task_context, :], 2)
        action_values_new = np.round(self.agent.Q_low[:, self.task_alien, :], 2)
        if self.mix_probs:
            print('Reward: {0} ({1})\n'
                  'Q_TS.     Old: {2}, RPE: {3}, New: {4}\n'
                  'Q_action. Old: {5}, RPE: {6}, New: {7}'.format(
                    str(np.round(reward, 2)), str(correct),
                    str(self.TS_values), str(np.round(self.agent.RPEs_high, 2)), str(TS_values_new),
                    str(self.action_values[:, action]), str(np.round(self.agent.RPEs_low, 2)),
                    str(action_values_new[:, action])))
        else:
            print('Reward: {0} ({1})\n'
                  'Q_TS.     Old: {2}, RPE: {3}, New: {4}\n'
                  'Q_action. Old: {5}, RPE: {6}, New: {7}'.format(
                    str(np.round(reward, 2)), str(correct),
                    str(self.TS_values[self.agent.TS]), str(np.round(self.agent.RPEs_high[self.agent.TS], 2)), str(TS_values_new[self.agent.TS]),
                    str(self.action_values[self.agent.TS, action]), str(np.round(self.agent.RPEs_low[self.agent.TS], 2)),
                    str(action_values_new[self.agent.TS, action])))
