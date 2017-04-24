function trial = practice_phase

global exp

%% Instructions
clearpict(exp.buffer.message)
preparestring('GAME OF COINS', exp.buffer.message, 0, 100)
preparestring('Try to collect all the coins!', exp.buffer.message, 0, -200)
drawpict(exp.buffer.message)
waitkeydown(inf, 71)   % Wait for spacebar

clearpict(exp.buffer.message)
preparestring('In this game, try to collect all the coins!', exp.buffer.message, 0, 300)
preparestring('The coins are hidden in magical boxes, like this one:', exp.buffer.message, 0, 250)
loadpict('stimuli/box.png', exp.buffer.message, 50, 0);
drawpict(exp.buffer.message)
waitkeydown(inf, 71)
preparestring('Can you open the box to see if it has a coin?', exp.buffer.message, 0, -200)
preparestring('Press any button to open the box!', exp.buffer.message, 0, -250)
drawpict(exp.buffer.message)
waitkeydown(inf)
loadpict('stimuli/box_coin.png', exp.buffer.message, 50, 0);
drawpict(exp.buffer.message)
waitkeydown(inf, 71)
preparestring('This box is magical! It contains a coin.', exp.buffer.message, 0, -300)
drawpict(exp.buffer.message)
waitkeydown(inf, 71)

%% Practice phase
% Phase 1: Left box is magical; deterministic; 7 trials
clearpict(exp.buffer.message)
preparestring('In the game, there will be two boxes, but only one is magical.', exp.buffer.message, 0, 200)
preparestring('Try finding the magical box. The left button opens the left box', exp.buffer.message, 0, 150)
preparestring('and the right button opens the right box.', exp.buffer.message, 0, 100)
drawpict(exp.buffer.message)
waitkeydown(inf, 71)

for trial = 1:exp.numb_of_trials.practice_left
    exp.better_box_left = 1;                                                % Left box is magical
    init_trial;                                                             % Initialize truth values
    
    pr_boxes;                                                               % Show boxes and let participant pick one
    pr_feedback('deterministic');                                           % Show reward or not
    
    rec_trial(trial);                                                       % Record trial data and save to file
    
    update_coin_counter(trial);
end
start_trial = trial + 1;

% Phase 2: Right box magical (det.; 5 trials); then left box (det.; 5 trials)
clearpict(exp.buffer.message)
preparestring('Do you have any questions?', exp.buffer.message, 0, 250)
drawpict(exp.buffer.message)
waitkeydown(inf, 71)
preparestring('Sometimes, the magical box switches sides!', exp.buffer.message, 0, 200)
preparestring('Try finding the magical box again!', exp.buffer.message, 0, 150)
drawpict(exp.buffer.message)
waitkeydown(inf, 71)
for trial = start_trial:(start_trial + exp.numb_of_trials.practice_switch1-1)
    exp.better_box_left = 0;                                                % Left box is magical
    init_trial;                                                             % Initialize truth values
    
    pr_boxes;                                                               % Show boxes and let participant pick one
    pr_feedback('deterministic');                                           % Show reward or not
    
    rec_trial(trial);                                                       % Record trial data and save to file
    
    update_coin_counter(trial);
end
start_trial = trial + 1;
for trial = start_trial:(start_trial + exp.numb_of_trials.practice_switch2-1)
    exp.better_box_left = 1;                                                % Left box is magical
    init_trial;                                                             % Initialize truth values
    
    pr_boxes;                                                               % Show boxes and let participant pick one
    pr_feedback('deterministic');                                           % Show reward or not
    
    rec_trial(trial);                                                       % Record trial data and save to file
    
    update_coin_counter(trial);
end
start_trial = trial + 1;

% Phase 3: Add stochastisity (identical to the true task)
clearpict(exp.buffer.message)
preparestring('Do you have any questions?', exp.buffer.message, 0, 250)
drawpict(exp.buffer.message)
waitkeydown(inf, 71)
preparestring('Sometimes, even the magical box does not have a coin.', exp.buffer.message, 0, 200)
preparestring('This makes it harder to find.', exp.buffer.message, 0, 150)
preparestring('Can you still find the magical box?', exp.buffer.message, 0, 100)
drawpict(exp.buffer.message)
waitkeydown(inf, 71)

for trial = start_trial:(start_trial + exp.numb_of_trials.practice_stochastic1-1)
    exp.better_box_left = 1;                                                % Left box is magical
    init_trial;                                                             % Initialize truth values
    
    pr_boxes;                                                               % Show boxes and let participant pick one
    pr_feedback('probabilistic');                                           % Show reward or not
    
    rec_trial(trial);                                                       % Record trial data and save to file
    
    update_coin_counter(trial);
end
start_trial = trial + 1;
for trial = start_trial:(start_trial + exp.numb_of_trials.practice_stochastic2-1)
    exp.better_box_left = 0;                                                % Right box is magical
    init_trial;                                                             % Initialize truth values
    
    pr_boxes;                                                               % Show boxes and let participant pick one
    pr_feedback('probabilistic');                                           % Show reward or not
    
    rec_trial(trial);                                                       % Record trial data and save to file
    
    update_coin_counter(trial);
end

%% Instructions for the real task
clearpict(exp.buffer.message)
preparestring('Do you have any questions?', exp.buffer.message, 0, 200)
drawpict(exp.buffer.message)
waitkeydown(inf, 71)
preparestring('Try your best to find as many coins as you can!', exp.buffer.message, 0, 100)
drawpict(exp.buffer.message)
waitkeydown(inf, 71)