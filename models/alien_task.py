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
        self.contexts = dict()
        for phase in self.phases:
            contexts = np.empty(0, dtype=int)
            n_trials = self.n_trials_per_phase[np.array(self.phases) == phase]
            len_block = self.block_lengths[np.array(self.phases) == phase]
            while len(contexts) < n_trials:
                if phase != '5RainbowSeason':
                    randomized_contexts = np.random.choice(range(self.n_contexts), size=self.n_contexts, replace=False)
                else:
                    randomized_contexts = [self.n_contexts + 1]
                new_block = np.concatenate([i * np.ones(len_block, dtype=int) for i in randomized_contexts])

                if phase == '5RainbowSeason':
                    attach_new_block = True
                elif len(contexts) > 0:
                    attach_new_block = contexts[-1] != new_block[0]

                if len(contexts) == 0:
                    contexts = new_block.copy()
                elif attach_new_block:
                    contexts = np.append(contexts, new_block)
            self.contexts[phase] = contexts

    def set_phase(self, new_phase):
        assert new_phase in self.phases
        self.phase = new_phase

    def prepare_trial(self, trial):
        self.context = self.contexts[self.phase][trial]
        # Get a random order of the four aliens every 4 trials
        if trial % self.n_aliens == 0:  # Every 4th trials, starting at 0
            self.shuffled_aliens = np.random.choice(range(self.n_aliens), size=self.n_aliens, replace=False)

    def present_stimulus(self, trial):
        self.alien = self.shuffled_aliens[trial % self.n_aliens]
        return np.array([self.context, self.alien])

    def produce_reward(self, action):
        if self.phase == '5RainbowSeason':
            return [np.nan, np.nan]  # no rewards during rainbow season
        else:
            reward = self.TS[self.context, self.alien, action]
            correct = reward > 1
            noised_reward = max(0, reward + np.round(np.random.normal(0, 0.5), 1))
            return [np.round(noised_reward, 2), correct]
