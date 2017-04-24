function pr_confidence

global exp

% start clock for confidence
start_time = clock;
exp.t_conf = drawpict(exp.buffer.mem_confidence);

% Record response
[exp.key_conf, exp.keytime_conf] = waitkeydown(exp.times.butterfly);

% Present response reminder after 5 sec
elapsed_time = [0 0 0 0 0 0];
while isempty(exp.key_conf)
    while elapsed_time(6) < exp.times.reminder/1000
        wait(100);
        elapsed_time = clock - start_time;
    end
    preparestring('PLEASE RESPOND', exp.buffer.mem_confidence, 0, 90);
    drawpict(exp.buffer.mem_confidence);
    break
end
if elapsed_time(6) > exp.times.reminder/1000
    [exp.key_conf, exp.key_conf] = waitkeydown(exp.times.butterfly-exp.times.reminder);
end

end