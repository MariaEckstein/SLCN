function pr_flower_and_butterfly

global exp

% Draw butterfly and flowers (as prepared in "init_trial")
start_time = clock;
exp.t = drawpict(exp.buffer.butterfly);

% Record response
[exp.key, exp.keytime] = waitkeydown(exp.times.butterfly);

% Present response reminder after 5 sec
elapsed_time = [0 0 0 0 0 0];
while isempty(exp.key)
    while elapsed_time(6) < exp.times.reminder/1000
        wait(100);
        elapsed_time = clock - start_time;
    end
    drawpict(exp.buffer.butterfly_reminder);
    break
end
if elapsed_time(6) > exp.times.reminder/1000
    [exp.key, exp.keytime] = waitkeydown(exp.times.butterfly-exp.times.reminder);
end


end
