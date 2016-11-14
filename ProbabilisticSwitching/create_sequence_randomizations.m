function create_sequence_randomizations(number_of_trials, possible_run_lengths)

% Determine how many rewards long each run will be (run = time during which
% one box is the better box; number of reward per run = uniform 7-23)

% possible_run_lengths = 1:15;
% number_of_trials = 150;

for sequence = 0:9   % Create 9 different sequences for different subjects
    run_length = [];
    for times = 1:ceil(0.75 * number_of_trials / mean(possible_run_lengths))
        run_length = [run_length, ...
            randsample(possible_run_lengths, length(possible_run_lengths), false)];
    end
    save(sprintf('Prerandomized sequences/run_length%i.mat', sequence), 'run_length');
end