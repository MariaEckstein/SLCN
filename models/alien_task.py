import numpy as np
import scipy.io as spio


class Task(object):
    def __init__(self, task_stuff, agent_id=np.nan):

        # Get parameters from task_stuff
        self.n_blocks = task_stuff['n_blocks']
        self.n_contexts = task_stuff['n_contexts']
        self.n_aliens = task_stuff['n_aliens']
        self.n_trials_per_alien = task_stuff['n_trials_per_alien']
        self.block_length = self.n_aliens * self.n_trials_per_alien
        self.n_trials = self.n_contexts * self.n_blocks * self.block_length
        self.n_actions = task_stuff['n_actions']

        # Info about current trial etc.
        self.shuffled_aliens = np.random.choice(range(4), size=4, replace=False)
        self.context = np.nan
        self.alien = np.nan
        self.TS = task_stuff['TS']

        # Create context order
        self.contexts = np.empty(0, dtype=int)
        for block in range(self.n_blocks):
            randomized_contexts = np.random.choice(range(self.n_contexts), size=self.n_contexts, replace=False)
            new_block = np.concatenate([i * np.ones(self.block_length, dtype=int) for i in randomized_contexts])
            self.contexts = np.append(self.contexts, new_block)

    def prepare_trial(self, trial):
        # Get current context
        self.context = self.contexts[trial]
        # Get a random order of the four aliens every 4 trials
        if trial % self.n_aliens == 0:
            self.shuffled_aliens = np.random.choice(range(self.n_aliens), size=self.n_aliens, replace=False)

    def present_stimulus(self, trial):
        # Return a context-alien stimulus
        self.alien = self.shuffled_aliens[trial % self.n_aliens]
        return np.array([self.context, self.alien])

    def produce_reward(self, action):
        reward = self.TS[self.context, self.alien, action]
        if reward == 0:
            return reward
        else:
            return reward + np.round(np.random.normal(0, 0.5), 1)
