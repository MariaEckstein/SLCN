function pr_feedback(trial)

global exp

%% Determine reward based on reward_sequence and reward_sequence_inc
trial_fly = exp.butterfly_sequence(trial);
reward_if_correct = exp.reward_sequence(exp.n_correct(trial_fly), trial_fly);
reward_if_incorrect = exp.reward_sequence_inc(exp.n_incorrect(trial_fly), trial_fly);
if isempty(exp.key) || (exp.key ~= exp.nkey.le && exp.key ~= exp.nkey.ri)
    if exp.show_points
        preparestring(strcat('Points: ', num2str(exp.earned_points)), exp.buffer.incorrect,0,300);
    end
    drawpict(exp.buffer.incorrect);                                             % Missing key presses are always wrong
elseif (exp.correct_response && reward_if_correct) || ...      % Correct butterfly picked; reward in 80% of trials, according to pre-randomized list
      (~exp.correct_response && reward_if_incorrect)       % Incorrect butterfly picked; reward in 20% of trials, according to pre-randomized list
    exp.earned_points = exp.earned_points + 1;
    if exp.show_points
        preparestring(strcat('Points: ', num2str(exp.earned_points)), exp.buffer.correct,0,300);
    end
    drawpict(exp.buffer.correct);
    exp.reward = 1;
else
    if exp.show_points
        preparestring(strcat('Points: ', num2str(exp.earned_points)), exp.buffer.incorrect,0,300);
    end
    drawpict(exp.buffer.incorrect);
end
wait(exp.times.reward);

end
