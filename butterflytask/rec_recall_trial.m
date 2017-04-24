function rec_recall_trial(trial)

global exp;

% Record info on memory test
% else
exp.RECALLdata.time_oldnew(trial)      = exp.t_oldnew;                              % Stimulus presentation time
exp.RECALLdata.ACC_oldnew(trial)       = exp.ACC_oldnew;

if ~isempty(exp.key_oldnew)
    exp.RECALLdata.key_oldnew(trial)   = exp.key_oldnew(1);
    exp.RECALLdata.keytime_oldnew(trial)= exp.keytime_oldnew(1);
    exp.RECALLdata.RT_oldnew(trial)    = exp.keytime_oldnew(1) - exp.t_oldnew;
    exp.RECALLdata.ACC_oldnew(trial)    = exp.ACC_oldnew(1);

else
    exp.RECALLdata.key_oldnew(trial)   = nan;
    exp.RECALLdata.keytime_oldnew(trial)= nan;
    exp.RECALLdata.RT_oldnew(trial)    = nan;
    exp.RECALLdata.ACC_oldnew(trial)    = nan;
end

exp.RECALLdata.time_conf(trial)      = exp.t_conf;  
if ~isempty(exp.key_conf)
    exp.RECALLdata.key_conf(trial)   = exp.key_conf(1);
    exp.RECALLdata.keytime_conf(trial)= exp.keytime_conf(1);
    exp.RECALLdata.RT_conf(trial)    = exp.keytime_conf(1) - exp.t_conf;
    exp.RECALLdata.ACC_conf(trial)   = exp.ACC_conf(1);
elseif exp.no_oldnew_ans
    exp.RECALLdata.key_conf(trial)   = nan;
    exp.RECALLdata.keytime_conf(trial)= nan;
    exp.RECALLdata.RT_conf(trial)    = nan;
    exp.RECALLdata.ACC_conf(trial)   = nan;
else
    exp.RECALLdata.key_conf(trial)   = nan;
    exp.RECALLdata.keytime_conf(trial)= nan;
    exp.RECALLdata.RT_conf(trial)    = nan;
    exp.RECALLdata.ACC_conf(trial)   = nan;
end

exp.key = [];
end_clock = clock;
exp.RECALLdata.end_time_h = end_clock(4);
exp.RECALLdata.end_time_m = end_clock(5);

%% Save results to file
result_file_name = [exp.results_filepath, sprintf('/Results/BUTTERFLY_%s.mat', num2str(exp.subj))];
save(result_file_name, 'exp');

end