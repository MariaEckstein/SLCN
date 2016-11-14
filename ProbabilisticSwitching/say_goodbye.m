function say_goodbye

global exp

%%
coins_won = num2str(nansum(exp.PROBSWITCHdata.reward));
clearpict(exp.buffer.message)
preparestring(['Great job! You won ', coins_won, ' coins in this game!'], exp.buffer.message, 0, 0)
drawpict(exp.buffer.message)
wait(300);
waitkeydown(inf)