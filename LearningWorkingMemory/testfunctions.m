% 
% clear all
% 
% debug = 1;
% 
% 
% Actions(1) = KbName('J');
% Actions(2) = KbName('K');
% Actions(3) = KbName('L');
% 
% 
% text{1} =  'Block ';
% text{2} = 'Take some time to identify the images for this block.';
% text{3} = '[Press a key to continue.]';
% text{4} = '';
% text{5} = '0';
% text{6} = 'End of block ';
% text{7} = 'End of this experiment!';
% text{8} = 'No valid answer';
% 
% acor = 2;
% 
% imaget = load(['imagesRLWM/images',num2str(12),'/image',num2str(1)]);
% imaget = imaget.X;
% 
% Screen('Preference', 'SkipSyncTests',1);
% if debug 
% screenRect = [0,0,1250,800]; % screen for debugging
% else
% screenRect = []; % full screen
% end
% [w, rect] = Screen('OpenWindow', 0, 0,screenRect,32,2);
% timing = [1 .5 1 .5];
% 
% [RT,code,cor]=singleTrial(w,rect,Actions,imaget,acor,timing);
% sca

%%
clear all

debug = 1;

Actions(1) = KbName('J');
Actions(2) = KbName('K');
Actions(3) = KbName('L');

Screen('Preference', 'SkipSyncTests',1);
if debug 
screenRect = [0,0,1250,800]; % screen for debugging
else
screenRect = []; % full screen
end
    SMat=zeros(300,300,3,2);
    % load stimuli and store them in matrix SMat for display
for i=1:2
    load(['imagesRLWM/TrainingImages/image',num2str(i)]);%%%load des stimuli
    SMat(:,:,:,i)=X;
end
    [w, rect] = Screen('OpenWindow', 0, 0,screenRect,32,2);
    InstructionsLSSt(w,rect,Actions,SMat);
