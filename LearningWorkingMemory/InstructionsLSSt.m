%%  RLWM + PST - new version 2014 (Anne Collins))

%%%% Anne Collins
%%%% Brown University
%%%% November 2014
%%%% anne_collins@brown.edu

%%%% RLWM task instructions


function    InstructionsLSSt(w,rect,Actions,SMat)
debug = 0;
load debugstate



% if debug
% [w, rect] = Screen('OpenWindow', 0, 0,[],32,2);
% end



wPtr=w;
Screen('TextSize', w, 24 );
Screen('TextFont',w,'Arial');
Screen('TextStyle',wPtr,0);
Screen('TextColor',wPtr,[255 255 255]);
beginningText1 = 'Working Memory Reinforcement Learning Task';
DrawFormattedText(wPtr,beginningText1,'center','center');
Screen(wPtr, 'Flip');
WaitSecs(3);

done = 1;

while done
% %****BEGINNING OF INSTRUCTIONS****
% %Instructions 1
anykey = 'Press any key to continue.';
%% Instructions1
text{1} = 'In this experiment, you will see an image on the screen.';
text{2} = 'Press a key to see an example on the next screen.';

textI = [text{1},'\n\n\n',text{2},'\n\n\n\n\n',anykey];
DrawFormattedText(wPtr,textI,'center','center');
Screen(wPtr, 'Flip');
KbWait([],3); %Waits for keyboard(any) press

imaget = SMat(:,:,:,1);
timing = [3 0 0 .5];
acor = 0;
singleTrial(w,rect,Actions,imaget,acor,timing);


%% Instructions 2
text{1} = 'Each time you see an image, you need to press a key:';
text{2} = 'you can see the three keys drawn under each image.';
text{3} = 'On the next screen, use your fingers to press one of the three keys.';

textI = [text{1},'\n\n\n',text{2},'\n\n\n',text{3},'\n\n\n\n\n\n\n',anykey];
Screen('TextSize', w, 24 );
DrawFormattedText(wPtr,textI,'center','center')
Screen(wPtr, 'Flip');

KbWait([],3); %Waits for keyboard(any) press

imaget = SMat(:,:,:,2);
timing = [5 0 0 .5];
singleTrial(w,rect,Actions,imaget,acor,timing);


%% Instructions 3
text{1} = 'You will need to decide quickly which key you want to press,';
text{2} = 'but no worries, you will have time to practice.';
text{3} = 'Try again on the next screen.';

textI = [text{1},'\n\n\n',text{2},'\n\n\n',text{3},'\n\n\n\n\n\n\n',anykey];
Screen('TextSize', w, 24 );
DrawFormattedText(wPtr,textI,'center','center')
Screen(wPtr, 'Flip');

KbWait([],3); %Waits for keyboard(any) press

imaget = SMat(:,:,:,2);
timing = [2 0 0 .5];
singleTrial(w,rect,Actions,imaget,acor,timing);





%% Instructions 4
text{1} = 'For each image, there is only one key that will win you points (1 / 1).';
text{2} = 'If you press any other key when you see that image, you will not receive points (0 / 1).';
text{3} = 'On the next screen, try finding the winning key!';

textI = [text{1},'\n\n\n',text{2},'\n\n\n',text{3},'\n\n\n\n\n\n\n',anykey];
Screen('TextSize', w, 24 );
DrawFormattedText(wPtr,textI,'center','center')
Screen(wPtr, 'Flip');

KbWait([],3); %Waits for keyboard(any) press

cor = 0;
acor = 2;
imaget = SMat(:,:,:,1);
timing = [2 .2 1 .5];
it = 0;
while cor<1 |it<5
[RT,code,cor]=singleTrial(w,rect,Actions,imaget,acor,timing);
it = it+1;
end

%% Instructions 5
text{1} = 'The winning key for each image can be different.';
text{2} = 'On the next screen, try finding the winning key for the other image!';

textI = [text{1},'\n\n\n',text{2},'\n\n\n\n\n\n\n',anykey];
Screen('TextSize', w, 24 );
DrawFormattedText(wPtr,textI,'center','center')
Screen(wPtr, 'Flip');

KbWait([],3); %Waits for keyboard(any) press

cor = 0;
imaget = SMat(:,:,:,2);
timing = [2 .2 1 .5];
acor = 3;
it = 0;
while cor<1 |it<5
[RT,code,cor]=singleTrial(w,rect,Actions,imaget,acor,timing);
it = it+1;
end


%% Instructions 6
text{1} = 'On each trial, try to press the winning key as quickly as possible.';
text{2} = 'If you do not choose quickly enough, you will not win points on this trial';
text{3} = 'On the next screen, try finding the winning keys for each image!';

textI = [text{1},'\n\n\n',text{2},'\n\n\n',text{3},'\n\n\n\n\n\n\n',anykey];
Screen('TextSize', w, 24 );
DrawFormattedText(wPtr,textI,'center','center')
Screen(wPtr, 'Flip');

KbWait([],3); %Waits for keyboard(any) press

timing = [2 .2 1 .5];
for iter = 1:10
imaget = SMat(:,:,:,1+rem(iter,2));
acor = 2+rem(iter,2);
[RT,code,cor]=singleTrial(w,rect,Actions,imaget,acor,timing);
end


%% Instructions 7
text{1} = 'You will first get a practice game, then you will play this game 12 times.';
text{2} = 'In each game, you will see new images,';
text{3} = 'and your job is to find the winning keys for each image!';

textI = [text{1},'\n\n\n',text{2},'\n\n\n',text{3},'\n\n\n\n\n\n\n',anykey];
Screen('TextSize', w, 24 );
DrawFormattedText(wPtr,textI,'center','center')
Screen(wPtr, 'Flip');
KbWait([],3); %Waits for keyboard(any) press

%% Propose to repeat instructions

text{1} = 'Ready to practice?';
text{2} = 'If not, you can press R to read the instructions again.';
text{3} = 'Otherwise, you can start the game by pressing any other key!';


textI = [text{1},'\n\n\n',text{2},'\n\n\n',text{3}];
Screen('TextSize', w, 24 );

DrawFormattedText(wPtr,textI,'center','center');
Screen(wPtr, 'Flip');

[keyIsDown, RT, keyCode] = KbCheck();
repeat = 0;
while repeat==0
    if keyIsDown 
        if keyCode(KbName('R'))
            repeat = 1;
        else
            repeat = -1;
        end
    end % if next key detected
    [keyIsDown, RT, keyCode] = KbCheck();
end % endless loop

if repeat == -1
    done = 0;
end
end

% textI = [text{23}];
% DrawFormattedText(wPtr,textI,'center','center');
% Screen(wPtr, 'Flip');
if debug
%Screen('CloseAll');
end
%****END OF INSTRUCTIONS****
