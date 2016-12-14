function final_screen

clearkeys;
clearpict(15);
b1 = 'You have finished the game!';
b2 = 'You did a great job!';
b3 = 'Press the space bar to finish.';
preparestring(b1, 15, 0, 250); 
preparestring(b2, 15, 0, 215); 
preparestring(b3, 15, 0, 140); 
drawpict(15);
waitkeydown(inf, 71);

end
    