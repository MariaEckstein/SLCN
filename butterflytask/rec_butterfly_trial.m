function rec_butterfly_trial(trial)

global exp;

%% Save trial variables to lists
% Record info on butterfly task
% if strcmp(butter_or_mem, 'butterfly')
    exp.BUTTERFLYdata.stimulus_time(trial)    = exp.t;                              % Stimulus presentation time
    exp.BUTTERFLYdata.butterfly(trial)        = exp.butterfly_sequence(trial);      % Which box was the better one? (1 => left better; 0 => right better)
    exp.BUTTERFLYdata.ACC(trial)              = exp.ACC(1);
    
    if ~isempty(exp.key)                                                             % If a keypress has been made
        exp.BUTTERFLYdata.key(trial)          = exp.key(1);                         % Which key has been pressed?
        exp.BUTTERFLYdata.keytime(trial)      = exp.keytime(1);                     % Keypress time
        exp.BUTTERFLYdata.RT(trial)           = exp.keytime(1) - exp.t;             % Time elapsed between cue onset (t) and time of key press (keytime)
        exp.BUTTERFLYdata.reward(trial)       = exp.reward;                         % 1 if participants gained a reward; 0 if no reward; nan if no response
    else                                                                             % If no keypress, evything is nan
        exp.BUTTERFLYdata.key(trial)          = nan;
        exp.BUTTERFLYdata.keytime(trial)      = nan;
        exp.BUTTERFLYdata.RT(trial)           = nan;
        exp.BUTTERFLYdata.reward(trial)       = nan;
    end

%% Save results to file
result_file_name = [exp.results_filepath, sprintf('/Results/BUTTERFLY_%s.mat', num2str(exp.subj))];
save(result_file_name, 'exp');
