import numpy as np
import scipy.io as spio


class Task(object):
    def __init__(self, task_stuff):

        # Get parameters from task_stuff
        self.phases = task_stuff['phases']
        self.n_trials_per_alien = task_stuff['n_trials_per_alien']
        self.n_blocks = task_stuff['n_blocks']
        self.n_aliens = task_stuff['n_aliens']
        self.block_lengths = self.n_aliens * self.n_trials_per_alien
        self.n_actions = task_stuff['n_actions']
        self.n_contexts = task_stuff['n_contexts']
        self.TS = task_stuff['TS']
        self.n_trials_per_phase = self.n_contexts * self.n_blocks * self.block_lengths

        # Info about current trial etc.
        self.shuffled_aliens = np.random.choice(range(4), size=4, replace=False)
        self.context = np.nan
        self.alien = np.nan
        self.phase = np.nan

        # Create context order
        self.contexts = np.empty(0, dtype=int)  # same sequence of contexts for initial learn & cloudy & refreshers
        # n_blocks_initial_learn = int(self.n_blocks[np.array(self.phases) == '1InitialLearning'])
        # for block in range(n_blocks_initial_learn):
        n_trials_initial_learn = self.n_trials_per_phase[np.array(self.phases) == '1InitialLearning']
        len_block_initial_learn = self.block_lengths[np.array(self.phases) == '1InitialLearning']
        while len(self.contexts) < n_trials_initial_learn :
            randomized_contexts = np.random.choice(range(self.n_contexts), size=self.n_contexts, replace=False)
            new_block = np.concatenate([i * np.ones(len_block_initial_learn, dtype=int) for i in randomized_contexts])
            if len(self.contexts) == 0:
                self.contexts = new_block.copy()
            elif self.contexts[-1] != new_block[0]:
                self.contexts = np.append(self.contexts, new_block)

    def set_phase(self, new_phase):
        assert new_phase in self.phases
        self.phase = new_phase

    def prepare_trial(self, trial):
        self.context = self.contexts[trial]
        # Get a random order of the four aliens every 4 trials
        if trial % self.n_aliens == 0:  # Every 4th trials, starting at 0
            self.shuffled_aliens = np.random.choice(range(self.n_aliens), size=self.n_aliens, replace=False)

    def present_stimulus(self, trial):
        self.alien = self.shuffled_aliens[trial % self.n_aliens]
        return np.array([self.context, self.alien])

    def produce_reward(self, action):
        reward = self.TS[self.context, self.alien, action]
        correct = reward > 1
        noised_reward = max(0, reward + np.round(np.random.normal(0, 0.5), 1))
        return [np.round(noised_reward, 2), correct]
