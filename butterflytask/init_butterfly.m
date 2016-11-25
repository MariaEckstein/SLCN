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
elseif exp.version == 4
    exp.le_flower_butterflies = [2 3];
    exp.ri_flower_butterflies = [1 4];
end


%% Configure Cogent
exp.numb_of_buffers = 15;
% Add cogent to the path
addpath(genpath('C:\Users\maria\MEGAsync\GSN\Wunderlich Lab\TrainingPlanningProject\Cogent2000v1.32'))
addpath(genpath('C:\Cogent2000v1.33'));
% Configure display & keyboard
config_display(1, 3, [0 0 0], [1 1 1], 'Helvetica', 40, exp.numb_of_buffers, 0);   % Configure display (0 = window mode; 5 = 1280x1024; [1 1 1] = white background; grey text; fontname; fontsize; nbuffers)
config_keyboard;

%% Load prerandomized sequences
% Load reward sequence so that 80% will be rewarded
reward_sequence_filename = sprintf('Prerandomized sequences/rewards%i', ...
    exp.version);
load(reward_sequence_filename);
exp.reward_sequence = reward_sequence;
% Load order of butterfly presentation
butterfly_sequence_filename = sprintf('Prerandomized sequences/butterfly%i', ...
    exp.version);
load(butterfly_sequence_filename);
exp.butterfly_sequence = butterfly_sequence;
% Set number of trials (practice and task)
exp.n_trials.practice = 8;
exp.n_trials.learning = 15; %155
exp.n_trials.test = 5; %32;
exp.n_trials.memory = 2 * exp.n_trials.learning;

%% Fix durations for everything in the experiment
exp.times.initial_fixation     =   250;
exp.times.butterfly            =  6000;    % Max response time
exp.times.display_choice       =  1000;    % How long is the selected box presented?
exp.times.reward               =  2000;
exp.times.iti                  =   500;

%% Set keys to select arrows and stimuli
exp.nkey.le          =  97;
exp.nkey.ri          =  98;

%% Positions on the screen
exp.p.butter_x   =    0;
exp.p.butter_y   = -250;
exp.p.flower_x   =  250;
exp.p.flower_y   =  250;

%% Get date and time
exp.BUTTERFLYdata.date = date;
start_clock = clock;
exp.BUTTERFLYdata.start_time_h = start_clock(4);
exp.BUTTERFLYdata.start_time_m = start_clock(5);

%% Set Variables
exp.correct_resp_counter = 1;
exp.wdir = cd;                                                              % Working directory
exp.results_filepath = exp.wdir;

%% Try to start cogent (else, wait for enter - to get matlab window)
try
    start_cogent;
catch
    input('Please press enter to continue')
    start_cogent;
end