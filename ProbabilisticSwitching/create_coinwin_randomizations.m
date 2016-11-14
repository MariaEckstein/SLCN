function create_coinwin_randomizations(number_of_trials, win_probability)

% Determine how rewards will be distributed if correct box is chosen

% number_of_trials = 150;
% win_probability = 0.75;

%% Prepare the sequences that will be used to create the long vector
% sequence of 8 ones, 2 zeros
seq_ceil = [ones(1, ceil(win_probability * 10)), ...
    zeros(1, floor(10 - win_probability * 10))];
% sequence of 7 ones, 3 zeros
seq_floor = [ones(1, floor(win_probability * 10)), ...
    zeros(1, ceil(10 - win_probability * 10))];

%% Put the long vector together (all trials)
for coinwin = 0:9   % Create 9 different sequences for different subjects
    coin_win = [];
    for times = 1:(20 + number_of_trials / (length(seq_ceil) + length(seq_floor)))
        coin_win = [coin_win, ...
            randsample(seq_ceil, length(seq_ceil), false), ...  % shuffle 8 ones & 2 zeros and add to the sequence
            randsample(seq_floor, length(seq_floor), false)];   % shuffle 7 ones & 3 zeros and add to the sequence
    end
    save(sprintf('Prerandomized sequences/coin_win%i.mat', coinwin), 'coin_win');
end
