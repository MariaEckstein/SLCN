function pr_feedback

global exp

%% Determine reward based on reward_sequence and reward_sequence_inc
if (isempty(exp.key) || (exp.key ~= exp.nkey.le && exp.key ~= exp.nkey.ri))               
    drawpict(exp.buffer.incorrect);                                             % Missing key presses are always wrong
elseif exp.correct_response && (exp.reward_sequence(exp.n_correct)) || ...      % Correct butterfly picked; reward in 80% of trials, according to pre-randomized list
      ~exp.correct_response && (exp.reward_sequence_inc(exp.n_incorrect))       % Incorrect butterfly picked; reward in 20% of trials, according to pre-randomized list
    drawpict(exp.buffer.correct);
    exp.reward = 1;
else
    drawpict(exp.buffer.incorrect);
end
wait(exp.times.reward);

end
