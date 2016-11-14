function choose_right_box_no_reward(trial)

global exp

init_trial;                                                             % Initialize truth values
pr_boxes;                                                               % Show boxes and let participant pick one
drawpict(exp.buffer.no_coin_right);
waitkeydown(exp.times.reward);
exp.reward = 0;
rec_trial(trial);                                                       % Record trial data and save to file
update_coin_counter(trial);
drawpict(exp.buffer.fixation);
waitkeydown(inf);