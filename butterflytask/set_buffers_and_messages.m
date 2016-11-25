function set_buffers_and_messages

global exp

%% Clear all buffers
for buffer = 1:exp.numb_of_buffers
    clearpict(buffer)
end

%% Name buffers
exp.buffer.butterfly                    = 1;   % Butterflies must have buffers 1:4 to make sure that randomization works!
exp.buffer.le_choice                    = 2;
exp.buffer.ri_choice                    = 3;
exp.buffer.correct                      = 9;
exp.buffer.incorrect                    =10;
exp.buffer.wrong_key                    =11;
exp.buffer.no_response                  =12;
exp.buffer.response_reminder            =13;

%% Load feedback into feedback buffers
preparestring('WRONG KEY (Use left and right arrow keys)', exp.buffer.wrong_key);
preparestring('NO RESPONSE', exp.buffer.no_response);
preparestring('CORRECT', exp.buffer.correct);
preparestring('INCORRECT', exp.buffer.incorrect);
preparestring('PLEASE RESPOND', exp.buffer.response_reminder);


end