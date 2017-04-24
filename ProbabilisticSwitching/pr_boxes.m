function pr_boxes

global exp

%% Draw boxes
exp.t = drawpict(exp.buffer.both_boxes);
[exp.key, exp.keytime] = waitkeydown(exp.times.boxes);
