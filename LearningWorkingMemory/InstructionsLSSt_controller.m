<<<<<<< HEAD
%%  RLWM + PST - new version 2014 (Anne Collins))

%%%% Anne Collins
%%%% Brown University
%%%% November 2014
%%%% anne_collins@brown.edu

%%%% RLWM task instructions


function    InstructionsLSSt_controller(w,rect,Actions,SMat)
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
anykey = ' ';

%% Instructions0 (intro story)
alien=imread('extra_images/alien','png');
controller=imread('extra_images/controller','png');
Screen('PutImage',wPtr,alien,[200 100 500 400]); % 350 x 350 > 300 x 300
Screen('PutImage',wPtr,controller, [600 100 1000 400]); % 400 x 300

textI = ['\n\n\n\n\n\n\n\n\n\n',anykey];
DrawFormattedText(wPtr,textI,'center','center');
Screen(wPtr, 'Flip');
RestrictKeysForKbCheck([32]); % wait for space
KbWait([],3); %Waits for keyboard(any) press
RestrictKeysForKbCheck([]);

%% Instructions1
% text{1} = 'In this game, you will see an image on the screen.';
% text{2} = 'Press any button to see an example on the next screen.';
% 
% textI = [text{1},'\n\n\n',text{2},'\n\n\n\n\n',anykey];
% DrawFormattedText(wPtr,textI,'center','center');
% Screen(wPtr, 'Flip');
% KbWait([],3); %Waits for keyboard(any) press
% 
imaget = SMat(:,:,:,1);
timing = [7 0 0 .5];
acor = 0;
% singleTrial(w,rect,Actions,imaget,acor,timing);


%% Instructions 2
text{1} = 'The aliens only use 3 buttons to communicate. Each time you see an picture, you need to press a button:';
text{2} = 'you can see the three buttons (left, middle, right) drawn under each picture.';
text{3} = 'On the next screen, press one of the three buttons on the controller.';

textI = [text{1},'\n\n\n',text{2},'\n\n\n',text{3},'\n\n\n\n\n\n\n',anykey];
Screen('TextSize', w, 24 );
DrawFormattedText(wPtr,textI,'center','center')
Screen(wPtr, 'Flip');
RestrictKeysForKbCheck([32]); % wait for space
KbWait([],3); %Waits for keyboard(any) press
RestrictKeysForKbCheck([]);

imaget = SMat(:,:,:,2);
timing = [7 0 0 .5]; % change from 3 to 7 sec
singleTrial(w,rect,Actions,imaget,acor,timing);


%% Instructions 3
text{1} = 'You will need to decide quickly which button you want to press,';
text{2} = 'but no worries, you will have time to practice.';
text{3} = ' ';

textI = [text{1},'\n\n\n',text{2},'\n\n\n',text{3},'\n\n\n\n\n\n\n',anykey];
Screen('TextSize', w, 24 );
DrawFormattedText(wPtr,textI,'center','center')
Screen(wPtr, 'Flip');

RestrictKeysForKbCheck([32]); % wait for space 
KbWait([],3); %Waits for keyboard(any) press
RestrictKeysForKbCheck([]);

imaget = SMat(:,:,:,2);
timing = [7 0 0 .5]; % change to 7 sec
singleTrial(w,rect,Actions,imaget,acor,timing);


%% Instructions 4
text{1} = 'The aliens only use one button for each picture.';
text{2} = 'If you press any other button, you will not receive points.';
text{4} = 'To show the aliens that youre learning their language,';
text{5} = 'press the correct button every time a picture comes up!';
text{3} = 'On the next screen, try finding the correct button.';
textI = [text{1},'\n\n\n',text{2},'\n\n\n',text{4},'\n\n\n',text{5},'\n\n\n',text{3},'\n\n\n\n\n\n\n',anykey];
Screen('TextSize', w, 24 );
DrawFormattedText(wPtr,textI,'center','center')
Screen(wPtr, 'Flip');

RestrictKeysForKbCheck([32]); % wait for space
KbWait([],3); %Waits for keyboard(any) press
RestrictKeysForKbCheck([]);

cor = 0;
acor = 2;
imaget = SMat(:,:,:,1);
timing = [7 .2 1 .5]; % change from 2 to 7 sec
it = 0;
while cor<1 |it<5
[RT,code,cor]=singleTrial(w,rect,Actions,imaget,acor,timing);
it = it+1;
end

%% Instructions 5
text{1} = ' ';
text{2} = 'On the next screen, try to find the correct button for a different picture!';

textI = [text{1},'\n\n\n',text{2},'\n\n\n\n\n\n\n',anykey];
Screen('TextSize', w, 24 );
DrawFormattedText(wPtr,textI,'center','center')
Screen(wPtr, 'Flip');

RestrictKeysForKbCheck([32]); % wait for space
KbWait([],3); %Waits for keyboard(any) press
RestrictKeysForKbCheck([]);
cor = 0;
imaget = SMat(:,:,:,2);
timing = [7 .2 1 .5]; % change to 7 sec
acor = 3;
it = 0;
while cor<1 |it<5
[RT,code,cor]=singleTrial(w,rect,Actions,imaget,acor,timing);
it = it+1;
end


%% Instructions 6
text{1} = 'Try to press the correct button as quickly as possible.';
text{2} = 'If you do not choose quickly enough, you will not win points in this round.';
text{3} = 'Try to find the correct buttons for two different pictures now!';

textI = [text{1},'\n\n\n',text{2},'\n\n\n',text{3},'\n\n\n\n\n\n\n',anykey];
Screen('TextSize', w, 24 );
DrawFormattedText(wPtr,textI,'center','center')
Screen(wPtr, 'Flip');

RestrictKeysForKbCheck([32]); % wait for space
KbWait([],3); %Waits for keyboard(any) press
RestrictKeysForKbCheck([]);

timing = [7 .2 1 .5]; % change to 7 sec
for iter = 1:10
imaget = SMat(:,:,:,1+rem(iter,2));
acor = 2+rem(iter,2);
[RT,code,cor]=singleTrial(w,rect,Actions,imaget,acor,timing);
end


%% Instructions 7
text{1} = 'First, you will first get a practice game, then you will play 10 levels of this game.';
text{2} = 'In each game, you will see new pictures,';
text{3} = 'and your job is to find the correct button for each image!';
text{4} = 'Sometimes you will see more than 3 pictures. Remember the aliens only use 3 buttons';
text{5} = 'to communicate, so the same button might be used for more than one picture.';

textI = [text{1},'\n\n\n',text{2},'\n\n\n',text{3},'\n\n\n',text{4},'\n\n\n',text{5},'\n\n\n\n\n\n\n',anykey];
Screen('TextSize', w, 24 );
DrawFormattedText(wPtr,textI,'center','center')
Screen(wPtr, 'Flip');

RestrictKeysForKbCheck([32]); % wait for space
KbWait([],3); %Waits for keyboard(any) press
RestrictKeysForKbCheck([]);

%% Propose to repeat instructions

text{1} = 'Ready to practice?';
text{2} = 'If not, you can press R to read the instructions again.';
text{3} = 'Otherwise, we can start the game.';


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
    RestrictKeysForKbCheck([32]);
    [keyIsDown, RT, keyCode] = KbCheck();
    RestrictKeysForKbCheck([]);
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
=======
%%  RLWM + PST - new version 2014 (Anne Collins))

%%%% Anne Collins
%%%% Brown University
%%%% November 2014
%%%% anne_collins@brown.edu

%%%% RLWM task instructions


function    InstructionsLSSt_controller(w,rect,Actions,SMat)
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
anykey = ' ';

%% Instructions0 (intro story)
alien=imread('extra_images/alien','png');
controller=imread('extra_images/controller','png');
Screen('PutImage',wPtr,alien,[200 100 500 400]); % 350 x 350 > 300 x 300
Screen('PutImage',wPtr,controller, [600 100 1000 400]); % 400 x 300

textI = ['\n\n\n\n\n\n\n\n\n\n',anykey];
DrawFormattedText(wPtr,textI,'center','center');
Screen(wPtr, 'Flip');
RestrictKeysForKbCheck([32]); % wait for space
KbWait([],3); %Waits for keyboard(any) press
RestrictKeysForKbCheck([]);

%% Instructions1
% text{1} = 'In this game, you will see an image on the screen.';
% text{2} = 'Press any button to see an example on the next screen.';
% 
% textI = [text{1},'\n\n\n',text{2},'\n\n\n\n\n',anykey];
% DrawFormattedText(wPtr,textI,'center','center');
% Screen(wPtr, 'Flip');
% KbWait([],3); %Waits for keyboard(any) press
% 
imaget = SMat(:,:,:,1);
timing = [3 0 0 .5];
acor = 0;
% singleTrial(w,rect,Actions,imaget,acor,timing);


%% Instructions 2
text{1} = 'The aliens only use 3 buttons to communicate. Each time you see an picture, you need to press a button:';
text{2} = 'you can see the three buttons (left, middle, right) drawn under each picture.';
text{3} = 'On the next screen, press one of the three buttons on the controller.';

textI = [text{1},'\n\n\n',text{2},'\n\n\n',text{3},'\n\n\n\n\n\n\n',anykey];
Screen('TextSize', w, 24 );
DrawFormattedText(wPtr,textI,'center','center')
Screen(wPtr, 'Flip');
RestrictKeysForKbCheck([32]); % wait for space
KbWait([],3); %Waits for keyboard(any) press
RestrictKeysForKbCheck([]);

imaget = SMat(:,:,:,2);
timing = [7 0 0 .5]; % change from 3 to 7 sec
singleTrial(w,rect,Actions,imaget,acor,timing);


%% Instructions 3
text{1} = 'You will need to decide quickly which button you want to press,';
text{2} = 'but no worries, you will have time to practice.';
text{3} = ' ';

textI = [text{1},'\n\n\n',text{2},'\n\n\n',text{3},'\n\n\n\n\n\n\n',anykey];
Screen('TextSize', w, 24 );
DrawFormattedText(wPtr,textI,'center','center')
Screen(wPtr, 'Flip');

RestrictKeysForKbCheck([32]); % wait for space 
KbWait([],3); %Waits for keyboard(any) press
RestrictKeysForKbCheck([]);

imaget = SMat(:,:,:,2);
timing = [7 0 0 .5]; % change to 7 sec
singleTrial(w,rect,Actions,imaget,acor,timing);


%% Instructions 4
text{1} = 'The aliens only use one button for each picture.';
text{2} = 'If you press any other button, you will not receive points.';
text{4} = 'To show the aliens that youre learning their language,';
text{5} = 'press the correct button every time a picture comes up!';
text{3} = 'On the next screen, try finding the correct button.';
textI = [text{1},'\n\n\n',text{2},'\n\n\n',text{4},'\n\n\n',text{5},'\n\n\n',text{3},'\n\n\n\n\n\n\n',anykey];
Screen('TextSize', w, 24 );
DrawFormattedText(wPtr,textI,'center','center')
Screen(wPtr, 'Flip');

RestrictKeysForKbCheck([32]); % wait for space
KbWait([],3); %Waits for keyboard(any) press
RestrictKeysForKbCheck([]);

cor = 0;
acor = 2;
imaget = SMat(:,:,:,1);
timing = [7 .2 1 .5]; % change from 2 to 7 sec
it = 0;
while cor<1 |it<5
[RT,code,cor]=singleTrial(w,rect,Actions,imaget,acor,timing);
it = it+1;
end

%% Instructions 5
text{1} = ' ';
text{2} = 'On the next screen, try to find the correct button for a different picture!';

textI = [text{1},'\n\n\n',text{2},'\n\n\n\n\n\n\n',anykey];
Screen('TextSize', w, 24 );
DrawFormattedText(wPtr,textI,'center','center')
Screen(wPtr, 'Flip');

RestrictKeysForKbCheck([32]); % wait for space
KbWait([],3); %Waits for keyboard(any) press
RestrictKeysForKbCheck([]);
cor = 0;
imaget = SMat(:,:,:,2);
timing = [7 .2 1 .5]; % change to 7 sec
acor = 3;
it = 0;
while cor<1 |it<5
[RT,code,cor]=singleTrial(w,rect,Actions,imaget,acor,timing);
it = it+1;
end


%% Instructions 6
text{1} = 'Try to press the correct button as quickly as possible.';
text{2} = 'If you do not choose quickly enough, you will not win points in this round.';
text{3} = 'Try to find the correct buttons for two different pictures now!';

textI = [text{1},'\n\n\n',text{2},'\n\n\n',text{3},'\n\n\n\n\n\n\n',anykey];
Screen('TextSize', w, 24 );
DrawFormattedText(wPtr,textI,'center','center')
Screen(wPtr, 'Flip');

RestrictKeysForKbCheck([32]); % wait for space
KbWait([],3); %Waits for keyboard(any) press
RestrictKeysForKbCheck([]);

timing = [2 .2 1 .5]; % change to 7 sec
for iter = 1:10
imaget = SMat(:,:,:,1+rem(iter,2));
acor = 2+rem(iter,2);
[RT,code,cor]=singleTrial(w,rect,Actions,imaget,acor,timing);
end


%% Instructions 7
text{1} = 'First, you will first get a practice game, then you will play 10 levels of this game.';
text{2} = 'In each game, you will see new pictures,';
text{3} = 'and your job is to find the correct button for each image!';
text{4} = 'Sometimes you will see more than 3 pictures. Remember the aliens only use 3 buttons';
text{5} = 'to communicate, so the same button might be used for more than one picture.';

textI = [text{1},'\n\n\n',text{2},'\n\n\n',text{3},'\n\n\n',text{4},'\n\n\n',text{5},'\n\n\n\n\n\n\n',anykey];
Screen('TextSize', w, 24 );
DrawFormattedText(wPtr,textI,'center','center')
Screen(wPtr, 'Flip');

RestrictKeysForKbCheck([32]); % wait for space
KbWait([],3); %Waits for keyboard(any) press
RestrictKeysForKbCheck([]);

%% Propose to repeat instructions

text{1} = 'Ready to practice?';
text{2} = 'If not, you can press R to read the instructions again.';
text{3} = 'Otherwise, we can start the game.';


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
    RestrictKeysForKbCheck([32]);
    [keyIsDown, RT, keyCode] = KbCheck();
    RestrictKeysForKbCheck([]);
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
>>>>>>> 0acae5fb6d80bfdcb27bc826cb7395dd92d45be6
