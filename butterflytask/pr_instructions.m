function pr_instructions(phase) % mode = {learn, train, test, mem} instructions

global exp;
upper_end = 300;
% 
if strcmp(phase, 'learn') % learning mode instructions
    % slide 0 - story slide
    clearkeys;
    clearpict(exp.buffer.instructions);
    loadpict('stims/storypic.jpg', ...
        exp.buffer.instructions,  0, 50, 434, 500);
    drawpict(exp.buffer.instructions);
    waitkeydown(inf,71);
    % slide 1
    clearkeys;
    clearpict(exp.buffer.instructions)    % refreshes the display buffer for instructions
    preparestring('In this game you need to guess which flower each butterfly wants to feed from.', ...
        exp.buffer.instructions, 0, 280);
    loadpict('stims/flowerf.jpg', ...
        exp.buffer.instructions, -250, 20, 130, 130);
    loadpict('stims/flowerg.jpg', ...
        exp.buffer.instructions, 250, 20, 130, 130);
    preparestring('You will get points if you guess the right flower!', ...
        exp.buffer.instructions, 0, -280);
    drawpict(exp.buffer.instructions);
    waitkeydown(inf, 71);
    
    % slide 2
    clearkeys;
    clearpict(exp.buffer.instructions);
    preparestring('Here is a hint: Each butterfly has a favourite flower, which it will pick more often than the other.', ...
        exp.buffer.instructions, 0, 210);
    preparestring('You will win more points if you figure out which flower each butterfly prefers.', ...
        exp.buffer.instructions, 0, 175);
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
    loadpict(butterfly1, exp.buffer.instructions, -255, 20, 130, 130);
    loadpict(butterfly2, exp.buffer.instructions, -85, 20, 130, 130);
    loadpict(butterfly3, exp.buffer.instructions, 85, 20, 130, 130);
    loadpict(butterfly4, exp.buffer.instructions, 255, 20, 130, 130);
    drawpict(exp.buffer.instructions);
    waitkeydown(inf, 71);
   
    
    % slide 3
    clearkeys;
    clearpict(exp.buffer.instructions);
    preparestring('Make sure to answer each trial. You will have 7 seconds', ...
        exp.buffer.instructions, 0, 105);
    preparestring('to make a choice, and will get a reminder after 5 seconds.',...
        exp.buffer.instructions, 0, 70);    
    preparestring('Use the left and right arrow keys to pick the flower you think the butterfly wants.', ...
        exp.buffer.instructions, 0, 35);
    preparestring( 'Let''s start with a quick practice.', exp.buffer.instructions, 0, -265);
    drawpict(exp.buffer.instructions);
    waitkeydown(inf, 71);
elseif strcmp(phase, 'train') % training mode with feedback
    clearkeys;
    clearpict(exp.buffer.instructions);
    preparestring
    preparestring('The practice round is over. Good job!',...
        exp.buffer.instructions, 0, 280);
    preparestring('The full game has 5 levels.',...
        exp.buffer.instructions, 0, 210);
    preparestring('This time, the butterflies will look different.',...
        exp.buffer.instructions, 0, 175);
    preparestring('You can take breaks after each level.',...
        exp.buffer.instructions, 0, 140);
    preparestring('The butterflies will have the same favorite flowers',...
        exp.buffer.instructions, 0, 70);
    preparestring('throughout the game.',...
        exp.buffer.instructions, 0, 35);
    drawpict(exp.buffer.instructions);
    waitkeydown(inf, 71);
    
elseif strcmp(phase, 'test') % testing mode instructions
    clearkeys;    
    clearpict(exp.buffer.instructions)    % refreshes the display buffer for instructions
    preparestring('Now the game will continue, and the butterflies still have the', ...
        exp.buffer.instructions, 0, 280);
    preparestring('same favorite flowers, but the computer wont tell you', exp.buffer.instructions, 0, 245);
    preparestring('if you are right or wrong.', exp.buffer.instructions, 0, 210);
    preparestring('It is still important that you try to give an answer on each trial.', ...
        exp.buffer.instructions, 0, 140);
    drawpict(exp.buffer.instructions);
    waitkeydown(inf, 71);
% 
else
% memory phase instructions
% slide 1
clearkeys;
clearpict(exp.buffer.instructions)    % refreshes the display buffer for instructions
preparestring('Now you will see some pictures of objects. You will have seen', ...
    exp.buffer.instructions, 0, upper_end);
preparestring('some of these objects from earlier in this game.', ...
    exp.buffer.instructions, 0, upper_end-35);
preparestring('On each trial you will be presented with an image.', ...
    exp.buffer.instructions, 0, upper_end-105);
preparestring('First you will decide whether the image is OLD or NEW.', ...
    exp.buffer.instructions, 0, upper_end-175);
preparestring('Press the "j" key if it''s an OLD image and', ...
    exp.buffer.instructions, 0, upper_end-210);
preparstring('he "l" key if it''s a NEW image.',...
    exp.buffer.instructions,0,upper_end-245);
preparestring('Press the "A" button to continue.', exp.buffer.instructions, 0, -85);
drawpict(exp.buffer.instructions);
waitkeydown(inf, 71);
% slide 2
clearkeys;
clearpict(exp.buffer.instructions);
preparestring('Then you will indicate how sure you are about that decision,', ...
    exp.buffer.instructions, 0, upper_end);
preparestring('By pressing the key for a number between (1) and (4).', ...
    exp.buffer.instructions, 0, upper_end-35);
preparestring('(1)', exp.buffer.instructions, -300, upper_end-110);
preparestring('(2)', exp.buffer.instructions, -100, upper_end-110);
preparestring('(3)', exp.buffer.instructions, 100, upper_end-110);
preparestring('(4)', exp.buffer.instructions, 300, upper_end-110);
loadpict('stims/certain.jpeg', exp.buffer.instructions,  -300, upper_end-210, 130, 130);
loadpict('stims/uncertain.jpg', exp.buffer.instructions,  300, upper_end-210, 130, 130);
preparestring('If you are completely certain, choose (1). If you are just guessing, choose (4).', ...
    exp.buffer.instructions, 0, -35);
preparestring('If you are somewhere in between, choose (2) for very sure, and (3) for pretty sure.', ...
    exp.buffer.instructions, 0, -70); 
preparestring('Press the "A" button to begin.', exp.buffer.instructions, 0, -155);
drawpict(exp.buffer.instructions);
waitkeydown(inf, 71);
end

end