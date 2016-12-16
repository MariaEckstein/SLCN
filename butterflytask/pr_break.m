function pr_break(trial)

global exp


%% Take a 1-minute break every 30 trials
if trial > 1 && mod(trial, 31) == 1
    drawpict(exp.buffer.break);
    waitkeydown(exp.times.break, 71);
    
    drawpict(exp.buffer.after_break);
    waitkeydown(30000, 71);
end


end