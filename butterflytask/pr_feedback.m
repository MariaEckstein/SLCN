function pr_feedback(version)

global exp

%% If participants pressed correct key, give reward if scheduled so in reward_sequence and if in the deterministic version
if exp.correct_response && (exp.reward_sequence(exp.correct_resp_counter) ...
        || strcmp(version, 'deterministic'))
    drawpict(exp.buffer.correct);
    exp.reward = 1;
else
    drawpict(exp.buffer.incorrect);
end
wait(exp.times.reward);

end
