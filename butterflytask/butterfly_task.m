function butterfly_task

clear all; 
global exp;

%% Define variables here during debugging
exp.subj = input('Subject ID: \n');

%% Prepare things
init_butterfly;
set_buffers_and_messages;

%% Pre-task practice
pr_instructions('learn');
for trial = 1:exp.n_trials.practice
    init_trial(trial, 'train', 'no_picture');                               % Initialize truth values; pick the butterfly to be presented
    
    pr_flower_and_butterfly;                                                % Show one butterfly and the flowers and let participant pick butterfly
    pr_choice(trial);                                                       % Show chosen flower and butterfly; determine if response was correct
    pr_feedback(trial, 'deterministic');                                    % Show CORRECT or INCORRECT plus random image
    
    rec_butterfly_trial(trial);                                             % Record trial data and save to file
end

%% Learning trials (with feedback)
pr_instructions('train');
exp.earned_points = 0;   % Reset point counter to 0
exp.n_correct = ones(1, 4);
exp.n_incorrect = ones(1, 4);
for trial = 1:exp.n_trials.learning
    init_trial(trial, 'exp', 'no_picture');                                 % Initialize truth values; pick the butterfly to be presented
    
    pr_flower_and_butterfly;                                                % Show one butterfly and the flowers and let participant pick butterfly
    pr_choice(trial);                                                       % Show chosen flower and butterfly; determine if response was correct
    pr_feedback(trial, 'stochastic');                                       % Show CORRECT or INCORRECT (no random image)
    
    rec_butterfly_trial(trial);                                             % Record trial data and save to file
    
    pr_break(trial);                                                        % Give a break after blocks of 30 trials
end

%% Test trials (without feedback)
pr_instructions('test'); 
exp.show_points = 0;
for trial = (1:exp.n_trials.test) + exp.n_trials.learning
    init_trial(trial, 'exp', 'no_picture');                                 % Initialize truth values; pick the butterfly to be presented
    
    pr_flower_and_butterfly;                                                % Show one butterfly and the flowers and let participant pick butterfly
    pr_choice(trial);                                                       % Show chosen flower and butterfly; determine if response was correct

    rec_butterfly_trial(trial);
    drawpict(exp.buffer.blank);                                             % show blank screen between testing phase trials
    wait(exp.times.iti);
end

exp.trial = trial;

%% Say goodbye
say_goodbye;
stop_cogent;
