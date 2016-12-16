function recall_task

global exp;

%% Define variables here during debugging
exp.subj = input('Subject ID: \n');

%% Prepare things
init_butterfly;
set_buffers_and_messages;

%% Add Cogent (MUST be an absolute path - relative paths do not stick after changing directory)
addpath(genpath('C:\Users\Amy\Desktop\butterflytask\Cogent2000v1.32'))
addpath('C:\Users\Amy\Desktop\butterflytask')

%% Surprise Memory test
pr_instructions('mem');
for trial = (1:exp.n_trials.memory/4) + exp.n_trials.learning + exp.n_trials.test
    init_mem(trial);
    
    rate_old_new;
    rate_confidence;
    
    rec_trial(trial, 'memory');

    pr_break(trial);                                                        % Give a break after blocks of 30 trials
end

%% Say goodbye
say_goodbye;

stop_cogent;