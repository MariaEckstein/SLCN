function pr_flower_and_butterfly

global exp

% Draw butterfly and flowers (as prepared in "init_trial")
exp.t = drawpict(exp.buffer.butterfly);

% Record response
[exp.key, exp.keytime] = waitkeydown(exp.times.butterfly);

end
