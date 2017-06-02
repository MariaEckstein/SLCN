function say_goodbye

clearkeys;
clearpict(16);
b1 = 'You have finished the game!';
b2 = 'You did a great job!';
preparestring(b1, 16, 0, 250); 
preparestring(b2, 16, 0, 215); 
drawpict(16);
waitkeydown(inf, 71);
    