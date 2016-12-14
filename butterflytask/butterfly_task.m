function butterfly_task

%clear all
global exp

%% Add Cogent (MUST be an absolute path - relative paths do not stick after changing directory)
addpath(genpath('C:\Users\Amy\Desktop\butterflytask\Cogent2000v1.32'))
addpath('C:\Users\Amy\Desktop\butterflytask')

%% Define variables here during debugging
exp.subj = input('Subject ID: \n');

%% Run the task
%%% Prepare things
init_butterfly;
set_buffers_and_messages;

%% Instructions 
exp.flyphase = '';

%% Pre-task practice
pr_instructions('learn');
exp.flyphase = 'learn';
for trial = 1:3%exp.n_trials.practice
    init_trial(trial, 'train', 'with_picture');                             % Initialize truth values; pick the butterfly to be presented
    
    pr_flower_and_butterfly;                                                % Show one butterfly and the flowers and let participant pick butterfly
    pr_choice(trial);                                                       % Show chosen flower and butterfly; determine if response was correct
    pr_feedback;                                                            % Show CORRECT or INCORRECT plus random image
    
    rec_trial(trial, 'butterfly');                                          % Record trial data and save to file
end

%pr_transition_learning

%% Learning trials (with feedback)
pr_instructions('train');
exp.flyphase = 'train';
for trial = 1:3%exp.n_trials.learning
    init_trial(trial, 'exp', 'with_picture');                               % Initialize truth values; pick the butterfly to be presented
    
    pr_flower_and_butterfly;                                                % Show one butterfly and the flowers and let participant pick butterfly
    pr_choice(trial);                                                       % Show chosen flower and butterfly; determine if response was correct
    pr_feedback;                                                            % Show CORRECT or INCORRECT (no random image)
    
    rec_trial(trial, 'butterfly');                                                       % Record trial data and save to file

    pr_break(trial);                                                        % Give a break after blocks of 30 trials
end

%pr_transition_test

%% Test trials (without feedback)
pr_instructions('test');
exp.flyphase = 'test';
for trial = 1:3%(1:exp.n_trials.test) + exp.n_trials.learning
    init_trial(trial, 'exp', 'without_picture');                            % Initialize truth values; pick the butterfly to be presented
    
    pr_flower_and_butterfly;                                                % Show one butterfly and the flowers and let participant pick butterfly
    pr_choice(trial);                                                       % Show chosen flower and butterfly; determine if response was correct

    rec_trial(trial, 'butterfly');
    drawpict(exp.buffer.blank);                                         % show blank screen between testing phase trials
end

% end screen
final_screen;

stop_cogent;
