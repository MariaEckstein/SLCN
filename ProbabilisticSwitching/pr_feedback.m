function pr_feedback(version)

global exp

%% Look up which key is the right one
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
% When participant pressed an allowed key
elseif exp.key(1) == exp.nkey.le || exp.key(1) == exp.nkey.ri
    % When participant pressed the correct key -> 75% chance of reward
    if exp.key(1) == better_key
        exp.ACC = 1;
        exp.right_box_chosen_counter = exp.right_box_chosen_counter + 1;
    end
    % Did participant make a switch?
    exp.made_switch = exp.switch_trial... 
        && exp.key(1) ~= exp.PROBSWITCHdata.key(exp.last_switch_trial);
    % Always reward when making switch to right box
    if exp.made_switch && (exp.coin_win(exp.right_box_chosen_counter)==0) % if made switch but coin_win is 0 for counter
        exp.next_coin_index = find(exp.coin_win(exp.right_box_chosen_counter:length(exp.coin_win))== 1, 1); % get next closest coin
        exp.coin_win(exp.right_box_chosen_counter)=1; % set counter value to 1
        exp.coin_win(exp.next_coin_index)=0; % switch values with old counter value 
    end
    if exp.key(1) == better_key && (exp.coin_win(exp.right_box_chosen_counter) ... 
            || strcmp(version, 'deterministic'))
        exp.reward = 1; 
        if exp.key(1) == exp.nkey.le
            drawpict(exp.buffer.coin_left);
        elseif exp.key(1) == exp.nkey.ri
            drawpict(exp.buffer.coin_right);
        end
        wait(exp.times.reward);
    % When participant pressed the wrong key, or the better key but bad luck -> No reward
    else
        if exp.key(1) == exp.nkey.le
            drawpict(exp.buffer.no_coin_left);
        elseif exp.key(1) == exp.nkey.ri
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
