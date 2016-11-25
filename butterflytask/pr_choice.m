function pr_choice(trial)

global exp

%% Check if participant chose a valid key and highlight choice if so
% When participants gave no response
if isempty(exp.key)
    drawpict(exp.buffer.no_response);
    wait(2 * exp.times.display_choice);
    
% When participant pressed a wrong key
elseif exp.key ~= exp.nkey.le && exp.key ~= exp.nkey.ri
    drawpict(exp.buffer.wrong_key);
    wait(2 * exp.times.display_choice);
    
% When participant picked the left flower, highlight the left flower
elseif exp.key == exp.nkey.le
    drawpict(exp.buffer.le_choice);
    wait(exp.times.display_choice);
    % if that was the correct choice, mark that
    if sum(exp.butterfly_sequence(trial) == exp.le_flower_butterflies)
        exp.correct_response = 1;
        exp.correct_resp_counter = exp.correct_resp_counter + 1;
    end
    
% When participant picked the right flower, highlight the right flower
elseif exp.key == exp.nkey.ri
    drawpict(exp.buffer.ri_choice);
    wait(exp.times.display_choice);
    % if that was the correct choice, mark that
    if sum(exp.butterfly_sequence(trial) == exp.ri_flower_butterflies)
        exp.correct_response = 1;
        exp.correct_resp_counter = exp.correct_resp_counter + 1;
    end
end

end
