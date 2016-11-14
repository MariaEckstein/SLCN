function init_prob_switch

global exp;

%% Configure Cogent
exp.numb_of_buffers = 15;
% Add cogent to the path
addpath(genpath('C:\Users\maria\MEGAsync\GSN\Wunderlich Lab\TrainingPlanningProject\Cogent2000v1.32'))
addpath(genpath('C:\Cogent2000v1.33'));
% Configure display & keyboard
config_display(1, 3, [1 1 1], [0 0 0], 'Helvetica', 40, exp.numb_of_buffers, 0);   % Configure display (0 = window mode; 5 = 1280x1024; [1 1 1] = white background; grey text; fontname; fontsize; nbuffers)
config_keyboard;

%% Set Variables
exp.wdir = cd;                                                              % Working directory
exp.results_filepath = exp.wdir;

%% Init data stuff that is needed by functions within first trial
exp.PROBSWITCHdata.switch_trial = [];
exp.PROBSWITCHdata.reward = [];

% Load run_length and coin_win randomizations (s1->r1; s2->r2, ..., s11->r1, s12->r12, ...)
run_length_filename = sprintf('Prerandomized sequences/run_length%i', ...
    mod(str2num(exp.subj), 10));
load(run_length_filename);
exp.run_length = run_length;
coin_win_filename = sprintf('Prerandomized sequences/coin_win%i', ...
    mod(str2num(exp.subj), 10));
load(coin_win_filename);
exp.coin_win = coin_win;
exp.right_box_chosen_counter = 1;
% Set number of trials (practice and task)
exp.numb_of_trials.prob_switch = 150;  % 150 trials plus instructions take exactly 10 minutes (13-year old)
exp.numb_of_trials.practice_left = 7;
exp.numb_of_trials.practice_switch1 = 6;
exp.numb_of_trials.practice_switch2 = 7;
exp.numb_of_trials.practice_stochastic1 = 12;
exp.numb_of_trials.practice_stochastic2 = 8;

% Keep track of which run we are currently at; start at 1
exp.run = 1;
% Flip a coin to find out which box will be better to start with
exp.better_box_left = randsample(0:1, 1);

% Fix durations for everything in the experiment
exp.times.initial_fixation     =   250;
exp.times.boxes                =  5000;    % Max response time
exp.times.box                  =   200;    % How long is the selected box presented?
exp.times.reward               =  1000;
exp.times.iti                  = ones(1, exp.numb_of_trials.prob_switch);                     % Inter-trial interval [0.3 0.7] with mean 0.5
while mean(exp.times.iti) > 501 || mean(exp.times.iti) < 499
    exp.times.iti = 300 + 400 * rand(1, exp.numb_of_trials.prob_switch + 50);
end

% Set keys to select arrows and stimuli
exp.nkey.le          =  97;
exp.nkey.ri          =  98;

%% Positions on the screen
exp.p.box_x   =  250;
exp.p.box_y   =  0;

%% Get date and time
exp.PROBSWITCHdata.date = date;
start_clock = clock;
exp.PROBSWITCHdata.start_time_h = start_clock(4);
exp.PROBSWITCHdata.start_time_m = start_clock(5);

%% Try to start cogent (else, wait for enter - to get matlab window)
try
    start_cogent;
catch
    input('Please press enter to continue')
    start_cogent;
end