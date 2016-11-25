function create_randomization_sequences(n_trials, n_randomizations)


%% Order of butterfly presentation
butterflies = [1:4,1:4];
for r = 1:n_randomizations
    butterfly_sequence = [];
    for times = 1:ceil(n_trials / length(butterflies))
        butterfly_sequence = [butterfly_sequence, ...
            randsample(butterflies, length(butterflies), false)];
    end
    save(sprintf('Prerandomized sequences/butterfly%i.mat', r), 'butterfly_sequence');
end

%% Order of CORRECT and INCORRECT feedback for correct responses (80%)
rewards = [ones(1, 8), zeros(1, 2)];
for r = 1:n_randomizations
    reward_sequence = [];
    for times = 1:ceil(n_trials / length(rewards))
        reward_sequence = [reward_sequence, ...
            randsample(rewards, length(rewards), false)];
    end
    save(sprintf('Prerandomized sequences/rewards%i.mat', r), 'reward_sequence');
end
