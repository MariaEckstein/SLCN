function init_trial(trial, train_or_exp, pic_no_pic)

global exp;


%% Initialize truth values
exp.key                 = [];
exp.keytime             = [];
exp.t_oldnew            = [];
exp.ACC_oldnew          = [];
exp.ACC_conf            = [];
exp.ACC_train           = [];
exp.ACC_test            = [];
exp.correct_response    = 0;
exp.reward              = 0;

%% Pick the right butterflies, flowers, and random images
% Decide which butterfly to present
if strcmp(train_or_exp, 'train')                                            % Training trials: Use the training butterflies
    exp.trial_butterfly = ...
        sprintf('stims/pracfly%i.jpg', exp.butterfly_sequence(trial));
elseif exp.version == 1 || exp.version == 3                                 % Experimental trials: Choise set of butterflies according to randomization group
    exp.trial_butterfly = ...
        sprintf('stims/butterflyw%i.jpg', exp.butterfly_sequence(trial));
elseif exp.version == 2 || exp.version == 0
    exp.trial_butterfly = ...
        sprintf('stims/butterflyw%i.jpg', exp.butterfly_sequence(trial) + 4);
end
% Pick the right flowers
if strcmp(train_or_exp, 'train')
    exp.trial_le_flower = 'stims/flowerprac1.jpg';
    exp.trial_ri_flower = 'stims/flowerprac2.jpg';
elseif exp.version == 1 || exp.version == 2
    exp.trial_le_flower = 'stims/flowerf.jpg';
    exp.trial_ri_flower = 'stims/flowerg.jpg';
elseif exp.version == 3 || exp.version == 0
    exp.trial_ri_flower = 'stims/flowerf.jpg';
    exp.trial_le_flower = 'stims/flowerg.jpg';
end
% Pick an image to be presented during feedback
if strcmp(train_or_exp, 'train')
    exp.image = sprintf('stims/P%02i.jpg', trial);
else
    exp.image = sprintf('stims/%s', char(exp.old_images(trial)));
end


%% Prepare buffers
% Clear necessary buffers and keys
clearkeys;
for buffer = [exp.buffer.butterfly, exp.buffer.butterfly_reminder, ...
        exp.buffer.le_choice, exp.buffer.ri_choice, ...
        exp.buffer.correct, exp.buffer.incorrect]
    clearpict(buffer)
end

% Create butterfly & flower buffer
for buffer = [exp.buffer.butterfly, exp.buffer.butterfly_reminder]
    loadpict(exp.trial_butterfly, buffer,  exp.p.butter_x, exp.p.butter_y);
    loadpict(exp.trial_le_flower, buffer, -exp.p.flower_x, exp.p.flower_y);
    loadpict(exp.trial_ri_flower, buffer,  exp.p.flower_x, exp.p.flower_y);
end
preparestring('PLEASE RESPOND', exp.buffer.butterfly_reminder);

% Create choice buffers
loadpict(exp.trial_butterfly, exp.buffer.le_choice,  exp.p.butter_x, exp.p.butter_y);
loadpict(exp.trial_le_flower, exp.buffer.le_choice, -exp.p.flower_x, exp.p.flower_y);
loadpict(exp.trial_butterfly, exp.buffer.ri_choice,  exp.p.butter_x, exp.p.butter_y);
loadpict(exp.trial_ri_flower, exp.buffer.ri_choice,  exp.p.flower_x, exp.p.flower_y);

% Create correct & incorrect buffers
if strcmp(pic_no_pic, 'with_picture')
    loadpict('stims/cor.jpg', exp.buffer.correct, exp.p.feedback_x, exp.p.feedback_y);
    loadpict('stims/blue.jpg', exp.buffer.correct, exp.p.image_x, exp.p.image_y);
    loadpict(exp.image, exp.buffer.correct, exp.p.image_x, exp.p.image_y);
    loadpict('stims/inc.jpg', exp.buffer.incorrect, exp.p.feedback_x, exp.p.feedback_y);
    loadpict('stims/red.jpg', exp.buffer.incorrect, exp.p.image_x, exp.p.image_y);
    loadpict(exp.image, exp.buffer.incorrect, exp.p.image_x, exp.p.image_y);
else
    loadpict('stims/cor.jpg', exp.buffer.correct, exp.p.feedback_x, exp.p.feedback_y);
    loadpict('stims/inc.jpg', exp.buffer.incorrect, exp.p.feedback_x, exp.p.feedback_y);
end


end
