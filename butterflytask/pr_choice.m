function pr_choice(trial)

global exp

%% Check if participant chose a valid key and highlight choice if so
% When participants gave no response
if isempty(exp.key)
    drawpict(exp.buffer.no_response);
    wait(2 * exp.times.display_choice);
    exp.ACC = nan;
    
% When participant pressed a wrong key
elseif exp.key ~= exp.nkey.le && exp.key ~= exp.nkey.ri
    drawpict(exp.buffer.wrong_key);
    wait(2 * exp.times.display_choice);
    exp.ACC = nan;
    
% When participant picked the left flower, highlight the left flower
elseif exp.key == exp.nkey.le
    drawpict(exp.buffer.le_choice);
    wait(exp.times.display_choice);
    % if that was the correct choice, mark that
    if sum(exp.butterfly_sequence(trial) == exp.le_flower_butterflies)
        exp.ACC = 1;
        exp.correct_response = 1;
    else
        exp.ACC = 0;
        exp.correct_response = 0;
    end
    
% When participant picked the right flower, highlight the right flower
elseif exp.key == exp.nkey.ri
    drawpict(exp.buffer.ri_choice);
    wait(exp.times.display_choice);
    % if that was the correct choice, mark that
    if sum(exp.butterfly_sequence(trial) == exp.ri_flower_butterflies)
        exp.ACC = 1;
        exp.correct_response = 1;
    else
        exp.ACC = 0;
        exp.correct_response = 0;
    end
end

% Count number of correct and incorrect responses (to move on in reward_sequence vector)
if exp.correct_response == 1
    exp.n_correct = exp.n_correct + 1;
else
    exp.n_incorrect = exp.n_incorrect + 1;
end


end
