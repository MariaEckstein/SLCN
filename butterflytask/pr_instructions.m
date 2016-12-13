function pr_instructions(phase) % mode = {learn, train, test, mem} instructions

upper_end = 300;
% 
if strcmp(phase, 'learn') % learning mode instructions
    % slide 1
    clearkeys;
    clearpict(14)    % refreshes the display buffer for instructions
    preparestring('In this game you will learn which flowers are preferred by different butterflies.', ...
        14, -22, upper_end);
    preparestring('On each trial you will see a butterfly and two flowers on the screen. You should', ...
        14, -10, upper_end-70);
    preparestring('use the left and right arrow keys to predict whether the butterfly prefers the left or', ...
        14, 0, upper_end-105);
    preparestring('the right flower.', 14, -365, upper_end-140);
    loadpict('stims/pracfly1.jpg', 14,  0, -200, 130, 130);
    loadpict('stims/flowerprac1.jpg', 14, -250, 20, 130, 130);
    loadpict('stims/flowerprac2.jpg', 14,  250, 20, 130, 130);
    preparestring('press <-',14,-250,-70);
    preparestring('press ->',14,250,-70);
    preparestring('Press the space bar to continue.', 14, -230, -310);
    drawpict(14);
    waitkeydown(inf, 71);
    
    % slide 2
    clearkeys;
    clearpict(14);
    preparestring('Each butterfly is able to feed from both flowers, but prefers one over the other.', ...
        14, -18, upper_end);
    preparestring('You should try to predict which flower the butterfly likes to feed from most of the', ...
        14, -6, upper_end-35);
    preparestring('time.', 14, -422, upper_end-70);
    preparestring('You will get feedback telling you if your predictions are correct. The word', ...
        14, -46, upper_end-140);
    preparestring('CORRECT or INCORRECT will be displayed on the screen along with a picture', ...
        14, -11, upper_end-175);
    preparestring('in a blue frame (CORRECT) or red frame (INCORRECT).', ...
        14, -136, upper_end-210);
    preparestring('Press the space bar to continue.', 14, -230, upper_end-280);
    drawpict(14);
    waitkeydown(inf, 71);
    
    % slide 3
    clearkeys;
    clearpict(14);
    preparestring('At first you will have to guess, but you will see all the butterflies multiple times,',...
        14, -10, upper_end);
    preparestring('and will improve your predictions as you play the game.',...
        14, -139, upper_end-35);
    preparestring('It is important that you give an answer on each trial. You have 7 seconds to', ...
        14, -31, upper_end-105);
    preparestring('make your responses and will be given a warning to respond after 5 seconds.', ...
        14, -18, upper_end-140);
    preparestring('If you don''t respond in time the screen will display TOO LATE.', ...
        14, -103, upper_end-175);
    preparestring('If you accidentally hold down a key while making responses the screen will', ...
        14, -33, upper_end-245);
    preparestring('display INVALID.', 14, -358, upper_end-280);
    preparestring( 'Let''s start with a quick practice.', 14, -235, -50);
    preparestring('Press the "s" key to begin.', 14, -270, -120);
    drawpict(14);
    waitkeydown(inf, 19);
elseif strcmp(phase, 'train') % training mode with feedback
    clearkeys;
    clearpict(14);
    preparestring('The practice round is over. The real trials will begin now.',...
        14, -103, upper_end-35);
    preparestring('Press the "s" key to begin.', 14, -270, upper_end-140);
    drawpict(14);
    waitkeydown(inf, 19);
    
elseif strcmp(phase, 'test') % testing mode instructions
    clearkeys;    
    clearpict(14)    % refreshes the display buffer for instructions
    preparestring('Now you will continue making flower selections, but you will not be told whether', ...
        14, 0, upper_end);
    preparestring('you are CORRECT or INCORRECT.', 14, -245, upper_end-35);
    preparestring('It is still important that you try to give an answer on each trial.', ...
        14, -105, upper_end-105);
    preparestring('Press the "s" key to begin.', 14, -270, upper_end-175);
    drawpict(14);
    waitkeydown(inf, 19);
% 
else
% memory phase instructions
% slide 1
clearkeys;
clearpict(14)    % refreshes the display buffer for instructions
preparestring('Now you will see some pictures of objects. You will have seen some of these', ...
    14, -39, upper_end);
preparestring('objects from earlier in this game.', ...
    14, -285, upper_end-35);
preparestring('On each trial you will be presented with an image.', ...
    14, -190, upper_end-105);
preparestring('First you will decide whether the image is OLD or NEW.', ...
    14, -157, upper_end-175);
preparestring('Press the left arrow key if it''s an OLD image and the right arrow key if it''s a NEW', ...
    14, -20, upper_end-210);
preparestring('image.', ...
    14, -433, upper_end-245);
preparestring('Press the space bar to continue.', ...
14, -250, -85);
drawpict(14);
waitkeydown(inf, 71);
% slide 2
clearkeys;
clearpict(14);
preparestring('Then you will indicate how sure you are about that decision,', ...
    14, -136, upper_end);
preparestring('By pressing the key for a number between (1) and (4).', ...
    14, -170, upper_end-35);
preparestring('(1)', 14, -300, upper_end-110);
preparestring('(2)', 14, -100, upper_end-110);
preparestring('(3)', 14, 100, upper_end-110);
preparestring('(4)', 14, 300, upper_end-110);
loadpict('stims/certain.jpeg', 14,  -300, upper_end-210, 130, 130);
loadpict('stims/uncertain.jpg', 14,  300, upper_end-210, 130, 130);
preparestring('If you are completely certain, choose (1). If you are just guessing, choose (4).', ...
    14, -42, -35);
preparestring('If you are somewhere in between, choose (2) for very sure, and (3) for pretty sure.', ...
    14, -18, -70); 
preparestring('Press the "s" key to begin.', 14, -270, -145);
drawpict(14);
waitkeydown(inf, 19);
end

end