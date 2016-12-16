function rate_old_new

global exp

% Present picture and old/new choice
exp.t_oldnew = drawpict(exp.buffer.mem_oldnew);
[exp.key_oldnew, exp.keytime_oldnew] = waitkeydown(exp.times.mem_test);

% Determine accuracy of decision
if ~isempty(exp.key_oldnew)
    if (strcmp(exp.oldnew, 'old') && exp.key_oldnew == exp.nkey.le) || ...
       (strcmp(exp.oldnew, 'new') && exp.key_oldnew == exp.nkey.ri)
        exp.ACC_oldnew = 1;
    elseif (strcmp(exp.oldnew, 'old') && exp.key_oldnew == exp.nkey.ri) || ...
       (strcmp(exp.oldnew, 'new') && exp.key_oldnew == exp.nkey.le)
        exp.ACC_oldnew = 0;
    else
        drawpict(exp.buffer.wrong_key);
        wait(2 * exp.times.display_choice);
        exp.ACC_oldnew = nan;
    end
else
    drawpict(exp.buffer.no_response);
    wait (2 * exp.times.display_choice);
    exp.ACC_oldnew = nan;
end
% Blank screen
drawpict(exp.buffer.blank);
wait(exp.times.iti);

end