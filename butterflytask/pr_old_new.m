function pr_old_new

global exp

% start clock for old/new
start_time = clock;
exp.t_oldnew = drawpict(exp.buffer.mem_oldnew);

% Record response
[exp.key_oldnew, exp.keytime_oldnew] = waitkeydown(exp.times.butterfly);

% Present response reminder after 5 sec
elapsed_time = [0 0 0 0 0 0];
while isempty(exp.key_oldnew)
    while elapsed_time(6) < exp.times.reminder/1000
        wait(100);
        elapsed_time = clock - start_time;
    end
    preparestring('PLEASE RESPOND', exp.buffer.mem_oldnew, 0, 90);
    drawpict(exp.buffer.mem_oldnew);
    break
end
if elapsed_time(6) > exp.times.reminder/1000
    [exp.key_oldnew, exp.keytime_oldnew] = waitkeydown(exp.times.butterfly-exp.times.reminder);
end

end