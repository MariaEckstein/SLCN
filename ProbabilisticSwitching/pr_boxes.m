function pr_boxes

global exp

%% Boxes
exp.t = drawpict(exp.buffer.both_boxes);
[exp.key, exp.keytime] = waitkeydown(exp.times.boxes);

end
