function init_mem(trial)

global exp


%% Initialize truth values
exp.key                 = [];
exp.keytime             = [];

%% Prepare buffers
% Clear buffers
clearpict(exp.buffer.mem_oldnew);
clearpict(exp.buffer.mem_confidence);
clearpict(exp.buffer.blank);
% Fill buffers
%%% Old/new rating
exp.image = sprintf('stims/%s', char(exp.mixed_images(trial)));
loadpict(exp.image, ...
    exp.buffer.mem_oldnew, exp.p.image_x, exp.p.image_y);
preparestring('Old                     ?                     New', ...
   exp.buffer.mem_oldnew, exp.p.feedback_x, exp.p.feedback_y);


%%% Confidence rating
loadpict(exp.image, ...
    exp.buffer.mem_confidence, exp.p.image_x, exp.p.image_y);
preparestring('(1)', exp.buffer.mem_confidence, -300, 260);
preparestring('(2)', exp.buffer.mem_confidence, -100, 260);
preparestring('(3)', exp.buffer.mem_confidence, 100, 260);
preparestring('(4)', exp.buffer.mem_confidence, 300, 260);
loadpict('stims/certain.jpeg', exp.buffer.mem_confidence,  -300, 160, 130, 130);
loadpict('stims/uncertain.jpg', exp.buffer.mem_confidence,  300, 160, 130, 130);

%%% "Blank" screen in-between
loadpict(exp.image, exp.buffer.blank, exp.p.image_x, exp.p.image_y);


%% Find out if image is old or new
if sum(strcmp(exp.old_images, exp.mixed_images(trial)))
    exp.oldnew = 'old';
else
    exp.oldnew = 'new';
end


end