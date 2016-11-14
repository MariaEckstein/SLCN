function choose_left_box_get_reward(trial)

global exp

init_trial;                                                             % Initialize truth values
pr_boxes;                                                               % Show boxes and let participant pick one
if exp.key == exp.nkey.le
    drawpict(exp.buffer.coin_left);
    wait(exp.times.reward);
else
    drawpict(exp.buffer.wrong_key);
    wait(exp.times.reward);
    pr_boxes;                                                               % Show boxes and let participant pick one
    if exp.key == exp.nkey.le
        drawpict(exp.buffer.coin_left);
        wait(exp.times.reward);
    else
        drawpict(exp.buffer.wrong_key);
        wait(exp.times.reward);
    end
end
exp.reward = 1;
rec_trial(trial);                                                       % Record trial data and save to file
update_coin_counter(trial);
drawpict(exp.buffer.fixation);
waitkeydown(inf);