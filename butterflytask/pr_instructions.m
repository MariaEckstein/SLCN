function pr_instructions(phase) % mode = {learn, train, test, mem} instructions

global exp;
upper_end = 300;
% 
if strcmp(phase, 'learn') % learning mode instructions
    % slide 0 - story slide
    clearkeys;
    clearpict(15);
    loadpict('stims/storypic.jpg', 15,  0, 50, 434, 500);
    preparestring(' ', 15, 0, -300);
    drawpict(15);
    waitkeydown(inf,71);
    % slide 1
    clearkeys;
    clearpict(15)    % refreshes the display buffer for instructions
    preparestring('In this game you need to guess which flower each butterfly wants to feed from.', ...
        15, 0, 280);
    loadpict('stims/flowerf.jpg', 15, -250, 20, 130, 130);
    loadpict('stims/flowerg.jpg', 15, 250, 20, 130, 130);
    preparestring('You will get points if you guess the right flower!', ...
        15, 0, -280);
    drawpict(15);
    waitkeydown(inf, 71);
    
    % slide 2
    clearkeys;
    clearpict(15);
    preparestring('Here is a hint: Each butterfly has a favourite flower, which it will pick more often than the other.', ...
        15, 0, 210);
    preparestring('You will win more points if you figure out which flower each butterfly prefers.', ...
        15, 0, 175);
    %%show four different butterflies
    if exp.version == 1 || exp.version == 3                                 % Experimental trials: Choise set of butterflies according to randomization group
        butterfly1 = 'stims/butterflyw1.jpg';
        butterfly2 = 'stims/butterflyw2.jpg';
        butterfly3 = 'stims/butterflyw3.jpg';
        butterfly4 = 'stims/butterflyw4.jpg';
    elseif exp.version == 2 || exp.version == 0
        butterfly1 ='stims/butterflyw5.jpg';
        butterfly2 = 'stims/butterflyw6.jpg';
        butterfly3 = 'stims/butterflyw7.jpg';
        butterfly4 = 'stims/butterflyw8.jpg';
    end
    loadpict(butterfly1, 15, -255, 20, 130, 130);
    loadpict(butterfly2, 15, -85, 20, 130, 130);
    loadpict(butterfly3, 15, 85, 20, 130, 130);
    loadpict(butterfly4, 15, 255, 20, 130, 130);
    drawpict(15);
    waitkeydown(inf, 71);
   
    
    % slide 3
    clearkeys;
    clearpict(15);
    preparestring('Make sure to answer each trial. You will have 7 seconds', ...
        15, 0, 105);
    preparestring('to make a choice, and will get a reminder after 5 seconds.',...
        15, 0, 70);    
    preparestring('Use the left and right arrow keys to pick the flower you think the butterfly wants.', ...
        15, 0, 35);
    preparestring( 'Let''s start with a quick practice.', 15, 0, -265);
    drawpict(15);
    waitkeydown(inf, 71);
elseif strcmp(phase, 'train') % training mode with feedback
    clearkeys;
    clearpict(15);
    preparestring
    preparestring('The practice round is over. Good job!',...
        15, 0, 280);
    preparestring('The full game has 5 levels.',...
        15, 0, 210);
    preparestring('This time, the butterflies will look different.',...
        15, 0, 175);
    preparestring('You can take breaks after each level.',...
        15, 0, 140);
    preparestring('The butterflies will have the same favorite flowers',...
        15, 0, 70);
    preparestring('throughout the game.',...
        15, 0, 35);
    drawpict(15);
    waitkeydown(inf, 71);
    
elseif strcmp(phase, 'test') % testing mode instructions
    clearkeys;    
    clearpict(15)    % refreshes the display buffer for instructions
    preparestring('Now the game will continue, and the butterflies still have the', ...
        15, 0, 280);
    preparestring('same favorite flowers, but the computer wont tell you', 15, 0, 245);
    preparestring('if you are right or wrong.', 15, 0, 210);
    preparestring('It is still important that you try to give an answer on each trial.', ...
        15, 0, 140);
    drawpict(15);
    waitkeydown(inf, 71);
% 
else
% memory phase instructions
% slide 1
clearkeys;
clearpict(15)    % refreshes the display buffer for instructions
preparestring('Now you will see some pictures of objects. You will have seen', ...
    15, 0, upper_end);
preparestring('some of these objects from earlier in this game.', ...
    15, 0, upper_end-35);
preparestring('On each trial you will be presented with an image.', ...
    15, 0, upper_end-105);
preparestring('First you will decide whether the image is OLD or NEW.', ...
    15, 0, upper_end-175);
preparestring('Press the "j" key if it''s an OLD image and', ...
    15, 0, upper_end-210);
preparstring('he "l" key if it''s a NEW image.',...
    15,0,upper_end-245);
preparestring('Press the "A" button to continue.', 15, 0, -85);
drawpict(15);
waitkeydown(inf, 71);
% slide 2
clearkeys;
clearpict(15);
preparestring('Then you will indicate how sure you are about that decision,', ...
    15, 0, upper_end);
preparestring('By pressing the key for a number between (1) and (4).', ...
    15, 0, upper_end-35);
preparestring('(1)', 15, -300, upper_end-110);
preparestring('(2)', 15, -100, upper_end-110);
preparestring('(3)', 15, 100, upper_end-110);
preparestring('(4)', 15, 300, upper_end-110);
loadpict('stims/certain.jpeg', 15,  -300, upper_end-210, 130, 130);
loadpict('stims/uncertain.jpg', 15,  300, upper_end-210, 130, 130);
preparestring('If you are completely certain, choose (1). If you are just guessing, choose (4).', ...
    15, 0, -35);
preparestring('If you are somewhere in between, choose (2) for very sure, and (3) for pretty sure.', ...
    15, 0, -70); 
preparestring('Press the "A" button to begin.', 15, 0, -155);
drawpict(15);
waitkeydown(inf, 71);
end

end