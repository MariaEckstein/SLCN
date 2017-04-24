function say_goodbye

global exp

%%
coins_won = num2str(nansum(exp.PROBSWITCHdata.reward));
clearpict(exp.buffer.message)
preparestring(['Great job! You found ', coins_won, ' coins in this game!'], ...
    exp.buffer.message, 0, 250)
loadpict('stimuli/leprachaun.png', ...
    exp.buffer.message, 0, -100);
drawpict(exp.buffer.message)
wait(300);
waitkeydown(inf)