function pr_break(trial)

global exp


%% Take a 1-minute break every 30 trials
break_every_x_trials = exp.numb_of_trials.prob_switch/3 + 1;
if (trial-exp.start_trial) > 1 && mod(trial, break_every_x_trials) == 1
    drawpict(exp.buffer.break);
    wait(exp.times.break);
    
    drawpict(exp.buffer.after_break);
    waitkeydown(30000);
end


end