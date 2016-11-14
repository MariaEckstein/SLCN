    function rec_trial(trial)

global exp;

%% Save trial variables to lists
exp.PROBSWITCHdata.stimulus_time(trial)    = exp.t;                              % Stimulus presentation time
exp.PROBSWITCHdata.better_box_left(trial)  = exp.better_box_left;                % Which box was the better one? (1 => left better; 0 => right better)
exp.PROBSWITCHdata.switch_trial(trial)     = exp.switch_trial;

if ~isempty(exp.key)                                                             % If a keypress has been made
    exp.PROBSWITCHdata.key(trial)          = exp.key(1);                         % Which key has been pressed?
    exp.PROBSWITCHdata.keytime(trial)      = exp.keytime(1);                     % Keypress time
    exp.PROBSWITCHdata.RT(trial)           = exp.keytime(1) - exp.t;             % Time elapsed between cue onset (t) and time of key press (keytime)
    exp.PROBSWITCHdata.reward(trial)       = exp.reward;                         % 1 if participants gained a reward; 0 if no reward; nan if no response
else                                                                             % If no keypress, evything is nan
    exp.PROBSWITCHdata.key(trial)          = nan;
    exp.PROBSWITCHdata.keytime(trial)      = nan;
    exp.PROBSWITCHdata.RT(trial)           = nan;
    exp.PROBSWITCHdata.reward(trial)       = nan;  
end

exp.key = [];
end_clock = clock;
exp.PROBSWITCHdata.end_time_h = end_clock(4);
exp.PROBSWITCHdata.end_time_m = end_clock(5);

%% Save results to file
result_file_name = [exp.results_filepath, sprintf('/Results/PROBSWITCH_%s.mat', exp.subj)];
save(result_file_name, 'exp');
