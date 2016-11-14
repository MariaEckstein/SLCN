function set_buffers_and_messages

global exp

%% Clear all buffers
for buffer = 1:exp.numb_of_buffers
    clearpict(buffer)
end

%% Name buffers
exp.buffer.text                         = 1;
exp.buffer.left_box                     = 2;
exp.buffer.right_box                    = 3;
exp.buffer.both_boxes                   = 4;
exp.buffer.coin_left                    = 5;
exp.buffer.coin_right                   = 6;
exp.buffer.fixation                     = 7;
exp.buffer.no_coin_left                 = 9;
exp.buffer.no_coin_right                =12;
exp.buffer.wrong_key                    =10;
exp.buffer.no_response                  =11;
exp.buffer.message                      =15;

%% Load box and coin pics into buffers
% both boxes
loadpict('stimuli/box.png', exp.buffer.both_boxes, exp.p.box_x, exp.p.box_y);
loadpict('stimuli/box.png', exp.buffer.both_boxes, -exp.p.box_x, exp.p.box_y);
% coin in left box
loadpict('stimuli/box.png', exp.buffer.coin_left, exp.p.box_x, exp.p.box_y);
loadpict('stimuli/box_coin.png', exp.buffer.coin_left, -exp.p.box_x, exp.p.box_y);
% coin in right box
loadpict('stimuli/box_coin.png', exp.buffer.coin_right, exp.p.box_x, exp.p.box_y);
loadpict('stimuli/box.png', exp.buffer.coin_right, -exp.p.box_x, exp.p.box_y);
% left box empty
loadpict('stimuli/box_empty.png', exp.buffer.no_coin_left, -exp.p.box_x, exp.p.box_y);
loadpict('stimuli/box.png', exp.buffer.no_coin_left, exp.p.box_x, exp.p.box_y);
% right box empty
loadpict('stimuli/box_empty.png', exp.buffer.no_coin_right, exp.p.box_x, exp.p.box_y);
loadpict('stimuli/box.png', exp.buffer.no_coin_right, -exp.p.box_x, exp.p.box_y);

%% Load feedback into feedback buffers
preparestring('wrong key! (Use left and right arrow keys)', exp.buffer.wrong_key);
preparestring('no response!', exp.buffer.no_response);

%% Load fixation cross into buffer
preparestring(' ', exp.buffer.fixation);

end