function pr_feedback(version)

global exp

%% Look up which box is the better one
if exp.better_box_left == 1
    better_key = exp.nkey.le;
elseif exp.better_box_left == 0
    better_key = exp.nkey.ri;
end

%% Determine accuracy and present feedback
% When participants gave no response
if isempty(exp.key)
    drawpict(exp.buffer.no_response);
    wait(2 * exp.times.reward);
% Check if participant pressed an allowed key
elseif exp.key == exp.nkey.le || exp.key == exp.nkey.ri
    % Was the better key pressed (= correct box selected)? -> safe this
    if exp.key == better_key
        exp.right_box_chosen_counter = exp.right_box_chosen_counter + 1;
    end
    %  Give reward as predetermined (or always in deterministic version)
    if exp.key == better_key && (exp.coin_win(exp.right_box_chosen_counter) ...
            || strcmp(version, 'deterministic'))
        exp.reward = 1;
        if exp.key == exp.nkey.le
            drawpict(exp.buffer.coin_left);
        elseif exp.key == exp.nkey.ri
            drawpict(exp.buffer.coin_right);
        end
        wait(exp.times.reward);
    % Was the worse key pressed or better key and bad luck? -> Give no reward
    else
        if exp.key == exp.nkey.le
            drawpict(exp.buffer.no_coin_left);
        elseif exp.key == exp.nkey.ri
            drawpict(exp.buffer.no_coin_right);
        end
        wait(exp.times.reward);
    end
% When participants pressed wrong key
else
    drawpict(exp.buffer.wrong_key);
    wait(2 * exp.times.reward);
end

end
