function butterfly_task

clear all
global exp


%% Define variables here during debugging
exp.subj = 1;%input('Subject ID: \n\n');

%% Run the task
%%% Prepare things
init_butterfly;
set_buffers_and_messages;

%%% Instructions
%pr_instructions;

%%% Pre-task practice
for trial = 1:exp.n_trials.practice
    init_trial(trial, 'train');                                             % Initialize truth values; pick the butterfly to be presented
    
    pr_flower_and_butterfly;                                                % Show one butterfly and the flowers and let participant pick butterfly
    pr_choice(trial);                                                       % Show chosen flower and butterfly; determine if response was correct
    pr_feedback('probabilistic');                                           % Show CORRECT or INCORRECT plus random image (TD!)
    
    rec_trial(trial);                                                       % Record trial data and save to file
end

%%% Learning trials (with feedback)
for trial = 1:exp.n_trials.learning
    init_trial(trial, 'exp');                                               % Initialize truth values; pick the butterfly to be presented
    
    pr_flower_and_butterfly;                                                % Show one butterfly and the flowers and let participant pick butterfly
    pr_choice(trial);                                                       % Show chosen flower and butterfly; determine if response was correct
    pr_feedback('probabilistic');                                           % Show CORRECT or INCORRECT plus random image (TD!)
    
    rec_trial(trial);                                                       % Record trial data and save to file
end

%%% Test trials (without feedback)
for trial = (1:exp.n_trials.test) + exp.n_trials.learning
    init_trial(trial, 'exp');                                               % Initialize truth values; pick the butterfly to be presented
    
    pr_flower_and_butterfly;                                                % Show one butterfly and the flowers and let participant pick butterfly
    pr_choice(trial);                                                       % Show chosen flower and butterfly; determine if response was correct
    
    rec_trial(trial);                                                       % Record trial data and save to file
end

%% Say goodbye
%say_goodbye;

stop_cogent;
