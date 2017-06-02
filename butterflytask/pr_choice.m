function pr_choice(trial)

global exp
clearkeys;

%% Check if participant chose a valid key and highlight choice if so
trial_fly = exp.butterfly_sequence(trial);

if ~isempty(exp.key)
    exp.key = exp.key(1);
    % Participant pressed left key
    if exp.key == exp.nkey.le
        buffer = exp.buffer.le_choice;
        correct_flys = exp.le_flower_butterflies;        
    % Participant pressed right key
    elseif exp.key == exp.nkey.ri
        buffer = exp.buffer.ri_choice;
        correct_flys = exp.ri_flower_butterflies;        
    % Participant pressed wrong key
    else
        buffer = exp.buffer.wrong_key;
        correct_flys = 99;
    end

    % Determine if choice was correct
    if any(trial_fly == correct_flys)
        exp.ACC = 1;
        exp.correct_response = 1;
    else
        exp.ACC = 0;
        exp.correct_response = 0;
    end

    % Update and show point counter
    if exp.show_points
        preparestring(strcat('Points: ', num2str(exp.earned_points)), buffer, exp.p.counter_x,300);
    end
    drawpict(buffer);
    wait(exp.times.display_choice);
    
% When participants gave no response
else
    if exp.show_points
        preparestring(strcat('Points: ', num2str(exp.earned_points)), exp.buffer.response_reminder,exp.p.counter_x,300);
        preparestring(strcat('Points: ', num2str(exp.earned_points)), exp.buffer.no_response,exp.p.counter_x,300);
    end
    drawpict(exp.buffer.no_response);
    wait(2 * exp.times.display_choice);
    exp.ACC = nan;
end

% Count number of correct and incorrect responses (to move on in reward_sequence vector)
if exp.ACC == 1  % correct response
    exp.n_correct(trial_fly) = exp.n_correct(trial_fly) + 1;
elseif exp.ACC == 0  % incorrect response
    exp.n_incorrect(trial_fly) = exp.n_incorrect(trial_fly) + 1;
end  % no response
