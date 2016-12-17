function init_butterfly

global exp;


%% Determine which butterfly likes which flower
% Determine randomization version based on subjID
n_randomizations = 4;
exp.version = mod(exp.subj, n_randomizations);
% Determine butterfly liking
if exp.version == 1
    exp.le_flower_butterflies = [1 2];
    exp.ri_flower_butterflies = [3 4];
elseif exp.version == 2
    exp.le_flower_butterflies = [1 3];
    exp.ri_flower_butterflies = [2 4];
elseif exp.version == 3
    exp.le_flower_butterflies = [1 4];
    exp.ri_flower_butterflies = [2 3];
elseif exp.version == 0
    exp.le_flower_butterflies = [2 3];
    exp.ri_flower_butterflies = [1 4];
end


%% Set number of trials (practice and task)
if exp.subj == 0   % Debugging: type in "0" for subj.id
    exp.n_trials.practice = 5;
    exp.n_trials.learning = 5;
    exp.n_trials.test = 5;
    exp.n_trials.memory = 2 * exp.n_trials.learning;
else
    exp.n_trials.practice = 8;
    exp.n_trials.learning = 120;
    exp.n_trials.test = 32;
    exp.n_trials.memory = 2 * exp.n_trials.learning;
end


%% Load prerandomized sequences
% Load reward sequence for correct choices (80% will be rewarded)
reward_sequence_filename = sprintf('Prerandomized sequences/rewards%i', ...
    exp.version);
load(reward_sequence_filename);
exp.reward_sequence = reward_sequences;
% Load reward sequence for incorrect choices (20% will be rewarded)
reward_inc_sequence_filename = sprintf('Prerandomized sequences/rewards_inc%i', ...
    exp.version);
load(reward_inc_sequence_filename);
exp.reward_sequence_inc = reward_sequences_inc;
% Load order of butterfly presentation
butterfly_sequence_filename = sprintf('Prerandomized sequences/butterfly%i', ...
    exp.version);
load(butterfly_sequence_filename);
exp.butterfly_sequence = butterfly_sequence;
% Load order of random images presented during the task
old_images_filename = sprintf('Prerandomized sequences/old_images%i', exp.version);
load(old_images_filename);
exp.old_images = old_images;
% Load mix of old and new images for the recall test
mixed_images_filename = sprintf('Prerandomized sequences/mixed_images%i', exp.version);
load(mixed_images_filename);
exp.mixed_images = mixed_images;


%% Fix durations for everything in the experiment
exp.times.initial_fixation     =   250;
exp.times.butterfly            =  7000;    % Max response time
exp.times.reminder             =  5000;    % Time of response reminder
exp.times.display_choice       =  1000;    % How long is the selected box presented?
exp.times.reward               =  2000;
exp.times.iti                  =   500;
exp.times.mem_test             = 10000;
exp.times.break                = 60000;    % 60000 msec = 60 sec = 1 minute


%% Set keys to select arrows and stimuli
exp.nkey.le          =  10;
exp.nkey.ri          =  12;


%% Positions on the screen
exp.p.butter_x   =    0;
exp.p.butter_y   = -250;
exp.p.flower_x   =  250;
exp.p.flower_y   =  250;
exp.p.feedback_x =    0;
exp.p.feedback_y =  150;
exp.p.image_x    =    0;
exp.p.image_y    = -100;


%% Get date and time
exp.BUTTERFLYdata.date = date;
start_clock = clock;
exp.BUTTERFLYdata.start_time_h = start_clock(4);
exp.BUTTERFLYdata.start_time_m = start_clock(5);


%% Set Variables
exp.n_correct = ones(1, 4);                                                 % Count of correct choices (separately for each butterfly)
exp.n_incorrect = ones(1, 4);                                               % Count of incorrect choices; used to find right spot in the pre-randomized reward sequence
exp.wdir = cd;                                                              % Working directory
exp.results_filepath = exp.wdir;


%% Configure Cogent
exp.numb_of_buffers = 16;
% Add cogent to the path
addpath(genpath('C:\Users\maria\Dropbox\NSFSLCN\Tasks\butterflytask\Cogent2000v1.32'));
addpath(genpath('C:\Users\prokofiev\Dropbox\NSFSLCN\Tasks\ButterflyTask\Cogent2000v1.32'));
addpath(genpath('C:\Users\Amy\Desktop\butterflytask\Cogent2000v1.32'))
addpath(genpath('C:\toolbox\Psychtoolbox'));
% Configure display & keyboard
config_display(1, 3, [0 0 0], [1 1 1], 'Helvetica', 28, exp.numb_of_buffers, 0);   % Configure display (0 = window mode; 5 = 1280x1024; [1 1 1] = white background; grey text; fontname; fontsize; nbuffers)
config_keyboard;
% Try to start cogent (else, wait for enter - to get matlab window)
try
    start_cogent;
catch
    input('Please press enter to continue')
    start_cogent;
end