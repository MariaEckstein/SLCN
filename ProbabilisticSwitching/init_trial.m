function init_trial

global exp;

%% Clear necessary buffers and keys
clearkeys;

%% Initialize truth values
exp.switch_trial        = 0;
exp.ACC                 = 0;
exp.key                 = [];
exp.keytime             = [];
exp.reward = 0;

end
