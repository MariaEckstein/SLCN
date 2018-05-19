import numpy as np


class CompetitionPhase(object):
    def __init__(self, comp_stuff, task_stuff):
        self.phases = comp_stuff['phases']
        self.n_blocks = comp_stuff['n_blocks']
        contexts = range(task_stuff['n_contexts'])
        context_pairs = [[cont_i, cont_j] for cont_i in contexts for cont_j in contexts if cont_i < cont_j]
        aliens = range(task_stuff['n_aliens'])
        alien_pairs = [[alien_i, alien_j] for alien_i in aliens for alien_j in aliens if alien_i < alien_j]
        items = range(task_stuff['n_actions'])
        item_pairs = [[item_i, item_j] for item_i in items for item_j in items if item_i < item_j]
        context_aliens = [[context, alien] for context in contexts for alien in aliens]
        context_alien_pairs = [[c_a_i, c_a_j] for c_a_i in context_aliens for c_a_j in context_aliens
                               if (c_a_i < c_a_j) and (c_a_i[0] == c_a_j[0])]  # same-season-pairs
        self.stimuli = {'season': context_pairs,
                        'alien-same-season': context_alien_pairs,
                        'alien': alien_pairs,
                        'item': item_pairs}
        self.n_trials = [self.n_blocks[phase] * len(self.stimuli[phase]) for phase in self.phases]
        self.switch_trials = [sum(self.n_trials[0:i]) for i in range(len(self.n_trials))]
        self.current_phase = self.phases[0]
        self.current_stimuli = self.stimuli[self.current_phase]
        self.shuffled_stimuli = np.nan
        self.block_trial = 0

    def prepare_trial(self, trial):
        # Look up current phase
        new_phase_index = np.argwhere(trial == np.array(self.switch_trials))
        if new_phase_index:
            self.current_phase = self.phases[new_phase_index]
            self.current_stimuli = self.stimuli[self.current_phase]
            self.block_trial = 0
        # Shuffle stimuli at the beginning of every block (3 per phase)
        if self.block_trial % len(self.current_stimuli) == 0:
            shuffled_indices = np.random.choice(range(len(self.current_stimuli)), size=len(self.current_stimuli), replace=False)
            self.shuffled_stimuli = [self.current_stimuli[i] for i in shuffled_indices]

    def present_stimulus(self, trial):
        return self.shuffled_stimuli[trial % len(self.shuffled_stimuli)]
