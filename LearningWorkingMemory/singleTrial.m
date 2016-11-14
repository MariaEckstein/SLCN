function [RT,code,cor]=singleTrial(w,rect,Actions,imaget,acor,timing)

prestime =timing(1);
AFB = timing(2);% action-FB delay
% FB presentation time 
FBprestime = timing(3);
% interstimulus interval
ISI= timing(4);

invalid = 'No valid answer';

red = [255 0 0];
green = [0 255 0];
%blue =[0 0 255];% 
gray = [127 127 127];
white = [255 255 255];
    
% 400 pixel rectangle for presentation. Adjust at will
p400 = min(rect(3:4))/4;
crectP = CenterRectOnPoint([0 0 p400 p400],rect(3)/2,rect(4)/2);% 3: width; 4: height
p200 = p400/3;
for k=1:3
    x = rect(3)/2 + (k-2)*1.5*p200;
    y= (rect(4)/2+ p400/2)/2 + rect(4)/2;
    keys{k}=CenterRectOnPoint([0 0 p200 p200],x,y);
end



% present it
Screen(w,'FillPoly',gray, [0 0;0 rect(4);rect(3) rect(4);rect(3) 0]);%%Ecrean total noir
Screen('PutImage',w,imaget,crectP);
for k = 1:3
    Screen(w,'FillRect',white,keys{k});
    Screen(w,'FrameRect',0,keys{k},2);
end
Screen('Flip', w);

% wait for a key press
tstart=GetSecs;
press=0;
RT=-1;
code=-1;
before = 0;
[kdown]=KbCheck;
before = find(kdown);
while GetSecs<tstart + prestime & press==0
    [kdown,secs,code]=KbCheck;

    if kdown == 1;
        press = 1;
        if press == 1;
            RT = secs-tstart;
            keycode = find(code==1);
            if length(keycode)==1 & intersect(keycode,Actions);  % if buttons
                code = find(keycode == Actions);
            else
                code=-1;
                press = 0; %% this is so other keys don't count as non answers
            end
        end
    end
end
if before
    code = -1;
    RT = -1;
    j1 = prestime;
else
j1 = AFB;
end
WaitSecs(j1);


if FBprestime>0
%deliver feedback if legal keypress ;occured
if code==1 | code==2 | code==3;
    if code==acor
        cor=1;
        word='1 / 1';
        couleur = green;
    else
        cor=0;
        word='0 / 1';%text{5};
        couleur = red;
    end


    Screen('TextFont',w,'Arial');
    Screen('TextSize', w, 32 );
    Screen(w,'FillPoly',gray, [0 0;0 rect(4);rect(3) rect(4);rect(3) 0]);%%Ecrean total noir
    Screen('TextColor',w,couleur);
    DrawFormattedText(w,word,'center','center'); %display +1 or +2 also

    %Screen(w,'DrawText',word,center(1)-150,center(2)-50,255);
    Screen('Flip', w);

else
    %no/illegal keypress: deliver no reward, encode as -1
    Screen('TextSize', w, 32 );
    Screen(w,'FillPoly',gray, [0 0;0 rect(4);rect(3) rect(4);rect(3) 0]);%%Ecrean total noir
    word = invalid;%,center(1)-150,center(2)-50,255);
    Screen('TextColor',w,red);
    DrawFormattedText(w,word,'center','center');
    Screen('Flip', w);

    code=-1;
    cor=-1;
    RT=-1;
end
WaitSecs(FBprestime);
end

Screen('TextColor',w,[255 255 255]);
Screen(w,'FillPoly',gray, [0 0;0 rect(4);rect(3) rect(4);rect(3) 0]);%%Ecrean total noir
Screen('TextSize', w, 32 );
DrawFormattedText(w,'+','center','center');
Screen('Flip', w);

WaitSecs(ISI);