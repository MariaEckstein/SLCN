function say_goodbye

clearkeys;
clearpict(16);
b1 = 'You have finished the game!';
b2 = 'You did a great job!';
b3 = 'Press the space bar to finish.';
preparestring(b1, 16, 0, 250); 
preparestring(b2, 16, 0, 215); 
preparestring(b3, 16, 0, 140); 
drawpict(16);
waitkeydown(inf, 71);

end
    