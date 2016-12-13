function rate_confidence

global exp

% Present picture and confidence of old/new rating
exp.t_conf = drawpict(exp.buffer.mem_confidence);
[exp.key_conf, exp.keytime_conf] = waitkeydown(exp.times.mem_test);
% Blank screen
if ~(exp.key_conf >= 28 && exp.key_conf <= 31)
    clearpict(exp.buffer.wrong_key);
    preparestring('WRONG KEY (Use number keys 1 to 4)', exp.buffer.wrong_key);
    drawpict(exp.buffer.wrong_key);
    wait(2 * exp.times.display_choice);
    exp.ACC_conf = nan;
else
    exp.ACC_conf = 1;
end

drawpict(exp.buffer.blank);
wait(exp.times.iti);

end