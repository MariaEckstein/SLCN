function rec_trial(trial, butter_or_mem)

global exp;

%% Save trial variables to lists
% Record info on butterfly task
if strcmp(butter_or_mem, 'butterfly')
    exp.BUTTERFLYdata.stimulus_time(trial)    = exp.t;                              % Stimulus presentation time
    exp.BUTTERFLYdata.butterfly(trial)        = exp.butterfly_sequence(trial);      % Which box was the better one? (1 => left better; 0 => right better)
    if strcmp(exp.flyphase, 'train')
        exp.BUTTERFLYdata.ACC_train(trial)        = exp.ACC(1);
    else
        exp.BUTTERFLYdata.ACC_test(trial)         = exp.ACC(1);
    end
    
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
    
% Record info on memory test
else
    exp.BUTTERFLYdata.time_oldnew(trial)      = exp.t_oldnew;                              % Stimulus presentation time
    exp.BUTTERFLYdata.ACC_oldnew(trial)       = exp.ACC_oldnew;

    if ~isempty(exp.key_oldnew)
        exp.BUTTERFLYdata.key_oldnew(trial)   = exp.key_oldnew(1);
        exp.BUTTERFLYdata.keytime_oldnew(trial)= exp.keytime_oldnew(1);
        exp.BUTTERFLYdata.RT_oldnew(trial)    = exp.keytime_oldnew(1) - exp.t_oldnew;
        exp.BUTTERFLYdata.ACC_oldnew(trial)    = exp.ACC_oldnew(1);

    else
        exp.BUTTERFLYdata.key_oldnew(trial)   = nan;
        exp.BUTTERFLYdata.keytime_oldnew(trial)= nan;
        exp.BUTTERFLYdata.RT_oldnew(trial)    = nan;
        exp.BUTTERFLYdata.ACC_oldnew(trial)    = nan;
    end
    
    exp.BUTTERFLYdata.time_conf(trial)      = exp.t_conf;  
    if ~isempty(exp.key_conf)
        exp.BUTTERFLYdata.key_conf(trial)   = exp.key_conf(1);
        exp.BUTTERFLYdata.keytime_conf(trial)= exp.keytime_conf(1);
        exp.BUTTERFLYdata.RT_conf(trial)    = exp.keytime_conf(1) - exp.t_conf;
        exp.BUTTERFLYdata.ACC_conf(trial)   = exp.ACC_conf(1);
    elseif exp.no_oldnew_ans
        exp.BUTTERFLYdata.key_conf(trial)   = nan;
        exp.BUTTERFLYdata.keytime_conf(trial)= nan;
        exp.BUTTERFLYdata.RT_conf(trial)    = nan;
        exp.BUTTERFLYdata.ACC_conf(trial)   = nan;
    else
        exp.BUTTERFLYdata.key_conf(trial)   = nan;
        exp.BUTTERFLYdata.keytime_conf(trial)= nan;
        exp.BUTTERFLYdata.RT_conf(trial)    = nan;
        exp.BUTTERFLYdata.ACC_conf(trial)   = nan;
    end
end

exp.key = [];
end_clock = clock;
exp.BUTTERFLYdata.end_time_h = end_clock(4);
exp.BUTTERFLYdata.end_time_m = end_clock(5);

%% Save results to file
result_file_name = [exp.results_filepath, sprintf('/Results/BUTTERFLY_%s.mat', num2str(exp.subj))];
save(result_file_name, 'exp');
