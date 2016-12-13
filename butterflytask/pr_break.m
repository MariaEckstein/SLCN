function pr_break(trial)

global exp


%% Take a 1-minute break every 30 trials
if trial > 1 && mod(trial, 30) == 1
    drawpict(exp.buffer.break);
    wait(exp.times.break);
    
    drawpict(exp.buffer.after_break);
    waitkeydown(30000);
end


end