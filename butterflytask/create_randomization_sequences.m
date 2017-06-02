function create_randomization_sequences(n_trials, n_randomizations)

%n_randomizations = 4;
%n_trials = 120;

%% Order of butterfly presentation
butterflies = [1:4, 1:4];
for r = 0:(n_randomizations-1)
    butterfly_sequence = [];
    for times = 1:ceil((n_trials+50) / length(butterflies))   % Needed for training and practice trials!
        butterfly_sequence = [butterfly_sequence, ...
            randsample(butterflies, length(butterflies), false)];
    end
    save(sprintf('Prerandomized sequences/butterfly%i.mat', r), 'butterfly_sequence');
end

%% Order of CORRECT and INCORRECT feedback for correct responses (80%)
rewards = [ones(1, 4), zeros(1, 1)];
for r = 0:(n_randomizations-1)   % 4 randomization versions
    reward_sequences = [];
    for b = 1:4   % 4 butterflies
        reward_sequence = [];
        for times = 1:ceil(n_trials / length(rewards))
            reward_sequence = [reward_sequence, ...
                randsample(rewards, length(rewards), false)];
        end
        reward_sequences(:,b) = reward_sequence;
    end
    save(sprintf('Prerandomized sequences/rewards%i.mat', r), 'reward_sequences');
end

%% Order of CORRECT and INCORRECT feedback for incorrect responses (20%)
rewards_inc = [ones(1, 1), zeros(1, 4)];
for r = 0:(n_randomizations-1)
    reward_sequences_inc = [];
    for b = 1:4   % 4 butterflies
        reward_sequence_inc = [];
        for times = 1:ceil(n_trials / length(rewards_inc))
            reward_sequence_inc = [reward_sequence_inc, ...
                randsample(rewards_inc, length(rewards_inc), false)];
        end
        reward_sequences_inc(:,b) = reward_sequence_inc;
    end
    save(sprintf('Prerandomized sequences/rewards_inc%i.mat', r), 'reward_sequences_inc');
end

%% Order of random images and distribution into old and new (learning phase)
% Get image names
n_images = 310;
images = cellstr('1.jpg');
for i = 2:n_images
    image = sprintf('%i.jpg', i);
    images = [images, image];
end
% 
for r = 0:(n_randomizations-1)
    images = randsample(images, length(images), false);
    first_half = 1:n_trials;
    second_half = (n_trials+1):(2*n_trials);
    old_images = images(first_half);
    new_images = images(second_half);
    save(sprintf('Prerandomized sequences/old_images%i.mat', r), 'old_images');
    save(sprintf('Prerandomized sequences/new_images%i.mat', r), 'new_images');

    %% Order of old and new images (surprise memory test)
    mixed_images = [];
    shuffled_old = randsample(old_images, length(old_images), false);
    shuffled_new = randsample(new_images, length(new_images), false);
    for chunk = 1:4:n_trials
        four_images = [shuffled_old(chunk:(chunk+3)), shuffled_new(chunk:(chunk+3))];
        shuffled_four_images = randsample(four_images, length(four_images), false);
        mixed_images = [mixed_images, shuffled_four_images];
    end
    save(sprintf('Prerandomized sequences/mixed_images%i.mat', r), 'mixed_images');
end


end