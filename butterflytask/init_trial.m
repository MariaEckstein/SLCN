function init_trial(trial, train_or_exp)

global exp;

%% Clear necessary buffers and keys
clearkeys;
for buffer = [exp.buffer.butterfly, exp.buffer.le_choice, exp.buffer.ri_choice]
    clearpict(buffer)
end

%% Initialize truth values
exp.ACC                 = nan;
exp.key                 = [];
exp.keytime             = [];
exp.correct_response    = 0;
exp.reward              = 0;

%% Decide which butterfly to present
if strcmp(train_or_exp, 'train')
    exp.trial_butterfly = ...
        sprintf('stims/pracfly%i.jpg', exp.butterfly_sequence(trial));
elseif exp.version == 1 || exp.version == 3
    exp.trial_butterfly = ...
        sprintf('stims/butterflyw%i.jpg', exp.butterfly_sequence(trial));
elseif exp.version == 2 || exp.version == 4
    exp.trial_butterfly = ...
        sprintf('stims/butterflyw%i.jpg', exp.butterfly_sequence(trial) + 4);
end
if strcmp(train_or_exp, 'train')
    exp.trial_le_flower = 'stims/flowerprac1.jpg';
    exp.trial_ri_flower = 'stims/flowerprac2.jpg';
elseif exp.version == 1 || exp.version == 2
    exp.trial_le_flower = 'stims/flowerf.jpg';
    exp.trial_ri_flower = 'stims/flowerg.jpg';
elseif exp.version == 3 || exp.version == 4
    exp.trial_ri_flower = 'stims/flowerf.jpg';
    exp.trial_le_flower = 'stims/flowerg.jpg';
end

%% Prepare buffers
% Butterfly & flower buffer
loadpict(exp.trial_butterfly, exp.buffer.butterfly,  exp.p.butter_x, exp.p.butter_y);
loadpict(exp.trial_le_flower, exp.buffer.butterfly, -exp.p.flower_x, exp.p.flower_y);
loadpict(exp.trial_ri_flower, exp.buffer.butterfly,  exp.p.flower_x, exp.p.flower_y);

% Choice buffers
loadpict(exp.trial_butterfly, exp.buffer.le_choice,  exp.p.butter_x, exp.p.butter_y);
loadpict(exp.trial_le_flower, exp.buffer.le_choice, -exp.p.flower_x, exp.p.flower_y);
loadpict(exp.trial_butterfly, exp.buffer.ri_choice,  exp.p.butter_x, exp.p.butter_y);
loadpict(exp.trial_ri_flower, exp.buffer.ri_choice,  exp.p.flower_x, exp.p.flower_y);


end
