function create_randomizations(n_trials, n_randomizations, win_probability, possible_run_lengths)

%% Determine how rewards will be distributed if correct box is chosen

% number_of_trials = 150;
% win_probability = 0.75;
% possible_run_lengths = 7:15

%%% Prepare the sequences that will be used to create the long vector
% sequence of 8 ones, 2 zeros
seq_ceil = [ones(1, ceil(win_probability * 10)), ...
    zeros(1, floor(10 - win_probability * 10))];
% sequence of 7 ones, 3 zeros
seq_floor = [ones(1, floor(win_probability * 10)), ...
    zeros(1, ceil(10 - win_probability * 10))];

%%% Put the long vector together (all trials)
for coinwin = 0:(n_randomizations-1)   % Create n_randomizations different sequences for different subjects
    coin_win = [];
    for times = 1:(20 + n_trials / (length(seq_ceil) + length(seq_floor)))
        coin_win = [coin_win, ...
            randsample(seq_ceil, length(seq_ceil), false), ...  % shuffle 8 ones & 2 zeros and add to the sequence
            randsample(seq_floor, length(seq_floor), false)];   % shuffle 7 ones & 3 zeros and add to the sequence
    end
    save(sprintf('Prerandomized sequences/coin_win%i.mat', coinwin), 'coin_win');
end


%% Determine run lengths
%%% How many rewards long will each run be (run = time during which
% one box is the better box)

for sequence = 0:(n_randomizations-1)
    run_length = [];
    for times = 1:ceil(0.75 * n_trials / mean(possible_run_lengths))
        run_length = [run_length, ...
            randsample(possible_run_lengths, length(possible_run_lengths), false)];
    end
    save(sprintf('Prerandomized sequences/run_length%i.mat', sequence), 'run_length');
end