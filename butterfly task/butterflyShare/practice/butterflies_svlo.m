% ********* delayFB.m
% *************************************************************
% Component of delay FB and context manipulation - test-version
% K. Foerde, 09/27/07
% *************************************************************************
% 1. Learn S-R task with correct/incorrect FB. Delay btw stim resp and FB
% is
%    delayed. 
% 2. Each trial consists of STIM secs of stimulus presentation, variable
%    amount of delay.
% 3. On each trial a novel scene is shown with the FB or just verbal FB .
% 4. FB is displayed for FB secs and ISI secs occurs before next trial. 
% 5. ntrials in each block and a break is given.
% 6. outdoor scenes are shown if CORRECT; indoor scenes if INCORRECT
% 7. This is SIMPLER version for PD and older CTRL
%**************************************************************************
clear all;

pwd
thePath.main=pwd;
[pathstr,curr_dir,ext,versn] = fileparts(pwd);
if ~strcmp(curr_dir,'dFB_practice')
    fprintf(['You must start the experiment from the dFB_practice directory. Go there and try again.\n']);
else
    thePath.scripts = fullfile(thePath.main, 'scripts');
    thePath.stim1 = fullfile(thePath.main, 'stims');
%     thePath.stim2 = fullfile(thePath.main, 'tibetan4');  
%     thePath.stim3 = fullfile(thePath.main, 'FB_images_4PD');      
%     thePath.data = fullfile(thePath.main, 'data');
    thePath.lists = fullfile(thePath.main, 'lists');

    %Add relevant paths for this experiment
    names = fieldnames(thePath);
    for f = 1:length(names)
        eval(['addpath(thePath.' names{f} ')']);
        fprintf(['added ' names{f} '\n']);
    end
end;

rand('state',sum(100*clock));  % seed the random number generator
subject_code = input('Enter subject code: ', 's');
condition = input('Is this practice (p)? ','s');

Verbal=input('Enter verbal FB (v or n): ', 's');

if strcmp(condition,'p'), 
FB_set='svlo';
% image_set='A';
% task_set=1;
list = 0;
version = 0;
startcycle = 1;
choice1='prac1';
choice2='prac2';
else
if strcmp(Verbal,'v'),
    FB_set='svlo';
else
    FB_set='svlo';
%     FB_set=input('Enter FB set (i, o, alli, allo, allmix, svlo): ', 's');
end;
stim_set=input('Butterfly (X, Y, z, w): ', 's');
choice1=input('Flower 1 (a-n): ', 's');
choice2=input('Flower 2 (a-n): ', 's');
% task_set=input('Enter task set (1 or 2; 0 for alli and allo): ');
task_set=1;
% image_set=input('Enter image set (A or B): ', 's');
list = input('List number (1 or 2): ');
% version = input('version number: ');
startcycle = input('Block number ? ');
end;

if strcmp(condition,'p'), 
infilename = sprintf('butterfly_%s_practice.mat',FB_set);
else
% infilename = sprintf('dFB_8_%s%d%d.mat',image_set,list,version);
% infilename = sprintf('tibet4_ord%dgrp%d%s_%s.mat',list,task_set,image_set,FB_set);
infilename = sprintf('tibet4_ord%d_%s.mat',list,FB_set);
% infilename = sprintf('tibet_4_ord%d_%s.mat',list,FB_set);
end;
load(infilename);
STIM=0; WARN=3; RESP_WIN=4; FB=2;  ISI=1; min_break_time=1; stim_off=0.25;

% define keys for each stimulus
c1_key=KbName('b');
c2_key=KbName('y');
c_key={'b' 'y'};
HideCursor;
% this next line is required to set up the psych toolbox
pixelSize = 32; 

realFMRI = input('Is this a scan session? 1 if yes, 0 if no: ');
% button_box = input('Are you using the button box? 1 if yes, 0 if no: ');

% timing of experiment
max_stim_duration=STIM+RESP_WIN;
warn_time=STIM+WARN;
show_selection=1;
warning_time=1;
stim_fb_interval=0.0;
fb_pres_time=FB;
% FMRI = 0; %change to 1 if you want base task to be part of loop
trigger_wait=3;
trigger='t';

present_fb=1;
if strcmp(condition,'p'),
ncycles=2;
ntrials=4;
else
ncycles=4;
ntrials=24;
end;
trial_counter=0;
fb_img_index=0; % counts pos and neg fb index to have no empty cells in test_image list for SM
posfb_img_index=0;
negfb_img_index=0;

% %%%%%%%%%%%% collect this info %%%%%%%
rt=zeros(1,ncycles*ntrials,5);
resp=cell(1,ncycles*ntrials);
trial_fb_image=cell(1,ncycles*ntrials);
shown_corr=zeros(1,ncycles*ntrials);
optimal_corr=zeros(1,ncycles*ntrials);
stim_shown=zeros(1,ncycles*ntrials);
stimtype_shown=zeros(1,ncycles*ntrials);
outc_shown=zeros(1,ncycles*ntrials);
delay_shown=zeros(1,ncycles*ntrials);
cat_select=zeros(1,ncycles*ntrials);
%% OSX %%
%set up screens
screens=Screen('Screens');
screenNumber=max(screens);
w=Screen('OpenWindow', screenNumber,0,[],32,2);
[wWidth, wHeight]=Screen('WindowSize', w);
black=BlackIndex(w); % Should equal 0.
white=WhiteIndex(w); % Should equal 255.
grayLevel=80;
% Screen('FillRect', w, grayLevel);
Screen('FillRect', w, black);
Screen('Flip', w);
HideCursor;
theFont='Arial';
FontSize=44;
Screen('TextSize',w,FontSize);
Screen('TextFont',w,theFont);

xcenter=wWidth/2;
ycenter=wHeight/2;
xstimsize=140;
ystimsize=140;
fbxstimsize=135;%200;
fbystimsize=135;
linesize=0;
yshift=50;

Lchoicerect=[xcenter-220 ycenter-260 xcenter-80 ycenter-120];
Rchoicerect=[xcenter+80 ycenter-260 xcenter+220 ycenter-120];
chosenrect=[xcenter-50 ycenter-220 xcenter+50 ycenter-120];
FBrect=[xcenter-fbxstimsize ycenter-fbystimsize xcenter+fbxstimsize ycenter+fbystimsize];
stimrect=[xcenter-xstimsize ycenter-ystimsize+yshift xcenter+xstimsize ycenter+ystimsize+yshift];
stimframerect=[xcenter-xstimsize-30 ycenter-ystimsize+yshift-30 xcenter+xstimsize+30 ycenter+ystimsize+yshift+30];

% [xcenter-xstimsize ycenter-ystimsize+yshift xcenter+xstimsize ycenter+ystimsize+yshift]

%%% create feedback graphical frames (around stimulus)
% fb_image_size=[340 460];
fb_image_size=[fbystimsize*2 fbxstimsize*2];
graphic = ones(fb_image_size(1)+30,fb_image_size(2)+30,3)*255;
a_frame=graphic;
a_frame(:,:,2)=0;
a_frame(:,:,3)=0;
b_frame=graphic;
b_frame(:,:,1)=0;
b_frame(:,:,2)=0;

black_graphic = ones(fb_image_size(1),fb_image_size(2),3)*255;
black_frame=black_graphic;
black_frame(:,:,:)=0;

stim_image_size=[ystimsize*2 xstimsize*2];
graphic = ones(stim_image_size(1)+30,stim_image_size(2)+30,3)*255;
c_frame=graphic;
c_frame(:,:,1)=0;
c_frame(:,:,3)=0;

tex1 = Screen('MakeTexture',w,a_frame);
tex2 = Screen('MakeTexture',w,b_frame);
tex3 = Screen('MakeTexture',w,black_frame);
tex4 = Screen('MakeTexture',w,c_frame);
% set up sounds for FB
snd('Open');
lo=sin(0:0.25:1000);
hi=sin(0:0.44:880);
% SND('Play',lo);
% SND('Play',hi);
sampling_rate=snd('DefaultRate');
% sound(lo)
% sound(hi)

% load in images for weather stims (scr cell for task and base_scr for baseline_
if strcmp(condition,'p'), 
    stim_images=cell(1,4);
for x=1:4, 
    stimPicname=sprintf('pracfly%d',x);
    stimPic=imread(stimPicname,'jpg');
    stimPicPtr{x}=Screen('MakeTexture',w,stimPic);
end;
else
stim_images=cell(1,4);
for x=1:4, 
    stimPicname=sprintf('butterfly%s%d',stim_set,x);
    stimPic=imread(stimPicname,'jpg');
    stimPicPtr{x}=Screen('MakeTexture',w,stimPic);
end;
end;

fname=sprintf('stims/flower%s',choice1); 
choice_image=imread(fname,'jpg');
choice_imagePic1=Screen('MakeTexture',w,choice_image);
fname=sprintf('stims/flower%s',choice2); 
choice_image=imread(fname,'jpg');
choice_imagePic2=Screen('MakeTexture',w,choice_image);


% new image loading for more fb list types
if strcmp(condition,'p') && strcmp(FB_set,'o'), 
    corr_image='blue frame';
    inc_image='red frame';
elseif strcmp(condition,'p') && strcmp(FB_set,'i'), 
% neg_dir='outdoor';
% pos_dir='indoor';
    corr_image='blue frame';
    inc_image='red frame';
elseif strcmp(FB_set,'o'),
%     pos_dir=sprintf('outdoor/grp%d%so',task_set,image_set);
%     neg_dir=sprintf('indoor/grp%d%si',task_set,image_set);
    corr_image='blue frame';
    inc_image='red frame';
elseif strcmp(FB_set,'i'),
%     pos_dir=sprintf('indoor/grp%d%si',task_set,image_set);
%     neg_dir=sprintf('outdoor/grp%d%so',task_set,image_set);
    corr_image='blue frame';
    inc_image='red frame';
else,
%     pos_dir=sprintf('indoor/grp%d%si',task_set,image_set);
%     neg_dir=sprintf('outdoor/grp%d%so',task_set,image_set);
    corr_image='blue frame';
    inc_image='red frame';
end;

d=clock;
startload=sprintf('%02.0f-%02.0f-%02.0f',d(4),d(5),d(6))
%reads in indoor and outdoor images for each trial
fb_images=cell(length(pos_images),2);
for x=1:length(pos_images),
    FBimg1=pos_images{x};
    FBimg{x,1}=pos_images{x};
    fname1=sprintf('stims/%s',FBimg1); 
    fb_images(x,1)={imread(fname1)};
    FBimg2=neg_images{x};
    FBimg{x,2}=neg_images{x};
    fname2=sprintf('stims/%s',FBimg2); 
    fb_images(x,2)={imread(fname2)};
end;
d=clock;
endload=sprintf('%02.0f-%02.0f-%02.0f',d(4),d(5),d(6))

% if strcmp(condition,'p') && strcmp(FB_set,'o'), 
% pos_dir='outdoor';
% neg_dir='indoor';
%     corr_image='OUTDOOR';
%     inc_image='INDOOR';
% elseif strcmp(condition,'p') && strcmp(FB_set,'i'), 
% neg_dir='outdoor';
% pos_dir='indoor';
%     corr_image='INDOOR';
%     inc_image='OUTDOOR';
% elseif strcmp(FB_set,'o'),
%     pos_dir=sprintf('outdoor/grp%d%so',task_set,image_set);
%     neg_dir=sprintf('indoor/grp%d%si',task_set,image_set);
%     corr_image='OUTDOOR';
%     inc_image='INDOOR';
% elseif strcmp(FB_set,'i'),
%     pos_dir=sprintf('indoor/grp%d%si',task_set,image_set);
%     neg_dir=sprintf('outdoor/grp%d%so',task_set,image_set);
%     corr_image='INDOOR';
%     inc_image='OUTDOOR';
% end;
% 
% d=clock;
% startload=sprintf('%02.0f-%02.0f-%02.0f',d(4),d(5),d(6))
% %reads in indoor and outdoor images for each trial
% fb_images=cell(length(pos_images),2);
% for x=1:length(pos_images),
%     FBimg1=pos_images{x};
%     FBimg{x,1}=pos_images{x};
%     fname1=sprintf('FB_images_4PD/%s/%s',pos_dir,FBimg1); 
%     fb_images(x,1)={imread(fname1)};
%     FBimg2=neg_images{x};
%     FBimg{x,2}=neg_images{x};
%     fname2=sprintf('FB_images_4PD/%s/%s',neg_dir,FBimg2); 
%     fb_images(x,2)={imread(fname2)};
% end;
% d=clock;
% endload=sprintf('%02.0f-%02.0f-%02.0f',d(4),d(5),d(6))

scr = cell(1,1);
base_scr = cell(1,1);

outcome_text={'    CORRECT  ' '  INCORRECT  ' };


%%%%%%%%%%%%%%%%%%%%%%%
%% set up input devices
numDevices=PsychHID('NumDevices');
devices=PsychHID('Devices');
if realFMRI==1,
    % for MRI inputDevice, 1st position is trigger and 2nd position is button box
    for n=1:numDevices,
        if (findstr(devices(n).transport,'USB') & findstr(devices(n).usageName,'Keyboard') & findstr(devices(n).manufacturer,'Apple')),
        elseif (findstr(devices(n).transport,'Bluetooth') & findstr(devices(n).usageName,'Keyboard') & findstr(devices(n).manufacturer,'Apple')),
        elseif (findstr(devices(n).transport,'ADB') & findstr(devices(n).usageName,'Keyboard') & findstr(devices(n).manufacturer,'Apple')),
            %         if (findstr(devices(n).transport,'USB') & findstr(devices(n).usageName,'Keyboard') & findstr(devices(n).manufacturer,'P.I. Engineering')),
            inputDevice(1)=n;
        end;
        if (findstr(devices(n).transport,'USB') & findstr(devices(n).usageName,'Keyboard') & (devices(n).productID==16385 | devices(n).vendorID==6171 | devices(n).totalElements==274)),
            inputDevice(2)=n;
        end;
    end;
    fprintf('Using Device #%d (%s)\n',inputDevice(2),devices(inputDevice(2)).product);
elseif exist('button_box')==1, 
   if button_box==1,
   for n=1:numDevices,
        if (findstr(devices(n).transport,'USB') & findstr(devices(n).usageName,'Keyboard') & findstr(devices(n).product,' USB Keypad')),
            inputDevice=[n n];
            break,
        elseif (findstr(devices(n).transport,'USB') & findstr(devices(n).usageName,'Keyboard') & (devices(n).productID==38936 | devices(n).vendorID==1444 | devices(n).totalElements==274)),
            inputDevice=[n n];
        end;
    end;
   end;
    fprintf('Using Device #%d (%s)\n',inputDevice(2),devices(inputDevice(2)).product);
else
    for n=1:numDevices,
            if (findstr(devices(n).transport,'USB') & findstr(devices(n).usageName,'Keyboard')),
                inputDevice=[n n];
                break,
            elseif (findstr(devices(n).transport,'Bluetooth') & findstr(devices(n).usageName,'Keyboard')),
                inputDevice=[n n];
                break,
            elseif (findstr(devices(n).transport,'ADB') & findstr(devices(n).usageName,'Keyboard')),
                inputDevice=[n n];
            end;
    end;
        fprintf('Using Device #%d (%s)\n',inputDevice(2),devices(n).product);
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% print instructions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Screen('TextSize',w,20);
%screen('CopyWindow',scr{16},w);

if strcmp(condition,'p'),
    if strcmp(FB_set,'svlo'),
        inst={{'In this game you will learn which flowers different butterflies like to feed from.'} ...
            {'On each trial you will see a butterfly and two flowers on the screen.'} ...
            {' '} ...
            {'You should use the '} ...
            {'BLUE button to choose the flower on the LEFT and the '} ...
            {'YELLOW button to choose the flower on the RIGHT.'} ...
            {' '} ...
            {'Each butterfly can feed from both flowers, but prefers one over the other.'}...
            {'Your job is to correctly predict which flower each butterfly likes to feed from most of the time.'}...
            {''}...
            {'You will be told if your predictions are correct. '}...
            {'The word CORRECT or INCORRECT will be displayed on the screen'}...
            {'along with a picture in a blue frame (CORRECT) or red frame (INCORRECT).'}...
            {'At first you will have to guess, but as the game goes on you should try your best to guess correctly.'}...
            {''}...
            {'In the game there will be short breaks. The butterfly preferences will remain the'}...
            {'same over these breaks.'}...
            {''}...
            {'It is important that you give an answer on each trial.'}...
            {'You have up to 4 seconds to make your responses, but try to go as FAST as you can.'}...
            {'If you don''t respond in time the screen will display TOO LATE.'}...
            {''}...
            {'Let''s start with a quick practice.'}...
            {'Press the "s" key to begin...'}};
        for x=1:size(inst,2),
        Screen('DrawText',w,inst{x}{:},100,100+x*30, white);
    end;
    Screen('Flip',w);
    end;
    noresp=1;
    while noresp,
            [keyIsDown,secs,keyCode] = KbCheck(inputDevice(1));
            if keyIsDown && noresp,
                noresp=0;
            end;
            WaitSecs(0.001);
    end;
    
else
%show this until trials starts
Screen('TextSize',w,60);
Screen('FillRect', w, black);
Screen('DrawText',w,'Experimenter - press t',xcenter-350,ycenter-20, white);
Screen('Flip', w);
    noresp=1;
    if realFMRI==1,
    while noresp,
            [keyIsDown,secs,keyCode] = KbCheck(inputDevice(1));
            if keyIsDown && noresp,
                if strcmp(KbName(keyCode),trigger)
                noresp=0;
                end;
            end;
            WaitSecs(0.001);
    end;
    else
    while noresp,
            [keyIsDown,secs,keyCode] = KbCheck(inputDevice(1));
            if keyIsDown && noresp,
                if strcmp(KbName(keyCode),trigger)
                noresp=0;
                end;
            end;
            WaitSecs(0.001);
    end;
    end;
end;
    WaitSecs(0.5);  % prevent key spillover        

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% write trial-by-trial data to a text logfile
d=clock;
logfile=sprintf('%s_dFB.log',subject_code);
fprintf('A log of this session will be saved to %s\n',logfile);
fid=fopen(logfile,'a');
if fid<1,
    error('could not open logfile!');
end;
fprintf(fid,'Started: %s %2.0f:%02.0f:%02.0f\n',date,d(4),d(5),d(6));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dd=clock;
fprintf(fid,'Started: %s %2.0f:%02.0f:%02.0f\n',date,dd(4),dd(5),dd(6));
absolute_anchor=GetSecs; % get absolute starting time anchor
noresp=0;  % set to zero initially

% loop through trials
for cycle=startcycle:ncycles,
    
noresp=1;
Screen('TextSize',w,60);
Screen('FillRect', w, black);
% Screen('DrawText',w,'''t'' to continue',xcenter-220,ycenter-20, white);
if strcmp(Verbal,'n'), reminder = [ corr_image ' = Correct'], 
Screen('DrawText',w,reminder',xcenter-300,ycenter-160, white);end;
Screen('DrawText',w,'GET READY!',xcenter-220,ycenter-20, white);
Screen('Flip', w);
%     while noresp,
%             [keyIsDown,secs,keyCode] = KbCheck(inputDevice(1));
%             if keyIsDown & noresp,
%                 if strcmp(KbName(keyCode),trigger)
%                 noresp=0;
%                 end;
%             end;
%             WaitSecs(0.001);
%     end;
%             WaitSecs(0.001);
%     Screen('TextSize',w,60);
%     Screen('FillRect', w, black);
%     Screen('DrawText',w,'+',xcenter-20,ycenter-20, white);
%     Screen('Flip', w);  
    anchor=GetSecs; % get absolute starting time anchor
%     WaitSecs(trigger_wait); % start manual so wait for first 4 measurements to be discarded

    for trial=1:ntrials, 
		trial_index=(cycle-1)*ntrials+trial;
%         while GetSecs - anchor < trialtimes(trial_index,1), end; %waits to synch beginning of trial with 'true' start

        %present image + questions
%         Screen('PutImage',w,stim_images{stim(trial_index,1)},[xcenter-xstimsize ycenter-ystimsize xcenter+xstimsize ycenter+ystimsize]); 
%         Screen('FrameRect',w,black,[xcenter-xstimsize ycenter-ystimsize xcenter+xstimsize ycenter+ystimsize],2);
% %         Screen('DrawText',w,'A or B ?',xcenter-112,ycenter-180, white);
%         Screen('Flip',w);
        stim_start_time=GetSecs;
		rt(1,trial_index,1)=stim_start_time-anchor;
        rt(1,trial_index,3)=stim_start_time-absolute_anchor;  
        stim_shown(1,trial_index)=stim(trial_index);
        stimtype_shown(1,trial_index)=stimtype(trial_index);
   
        %ask for response
        Screen('TextSize',w,60);        
        stimPtr=stimPicPtr{stim(trial_index,1)};
        Screen('DrawTexture',w,stimPtr,[],stimrect,[],[],[],[]);
%         Screen('FrameRect',w,[0 250 100],stimrect,5);
%         Screen('FrameRect',w,[0 250 100],Lchoicerect,5);
%         Screen('FrameRect',w,[0 250 100],Rchoicerect,5);
        Screen('DrawTexture',w,choice_imagePic1,[],Lchoicerect,[],[],[],[]);
        Screen('DrawTexture',w,choice_imagePic2,[],Rchoicerect,[],[],[],[]);
        Screen('DrawText',w,'     ?',xcenter-112,ycenter-250, white);
        Screen('Flip',w);        
        resp_start_time=GetSecs;        
      
        % to collect responses and RTs in OSX
        noresp=1;
        
        if WARN==0,        warned=1;    else warned=0; end;
        
        % warn to respond
        while (GetSecs - resp_start_time < max_stim_duration) && noresp,   
        [keyIsDown,secs,keyCode] = KbCheck(inputDevice(2));
%             if keyIsDown && noresp &&
%             (strcmp(KbName(keyCode),trigger)~=1), % only need for fMRI
            if keyIsDown && noresp,% && (strcmp(KbName(keyCode),trigger)~=1), 
                tmp=KbName(keyCode);
%                 if sum(strcmp({tmp(1)},c_key)>0),
%                 resp(1,trial_index)={tmp(1)};
                if(keyCode(c1_key) || keyCode(c2_key)),
                resp(1,trial_index)={tmp};
                rt(1,trial_index,2)=GetSecs-resp_start_time; 
                noresp=0;
                end;
            WaitSecs(.001);
            end;	
            
           if  ~warned && (GetSecs - resp_start_time > warn_time)
                stimPtr=stimPicPtr{stim(trial_index,1)};
        Screen('DrawTexture',w,stimPtr,[],stimrect,[],[],[],[]);
%         Screen('FrameRect',w,[0 250 100],stimrect,5);
        Screen('DrawText',w,'PLEASE RESPOND',xcenter-242,ycenter-370, white);
%         Screen('FrameRect',w,[0 250 100],Lchoicerect,5);
%         Screen('FrameRect',w,[0 250 100],Rchoicerect,5);
        Screen('DrawTexture',w,choice_imagePic1,[],Lchoicerect,[],[],[],[]);
        Screen('DrawTexture',w,choice_imagePic2,[],Rchoicerect,[],[],[],[]);
        Screen('DrawText',w,'     ?',xcenter-112,ycenter-250, white);
%         Screen('DrawText',w,'A or B ?',xcenter-112,ycenter-180, white);
        Screen('Flip',w);     
           warned=1;
           end;    
            
        WaitSecs(0.001);
        end;
        WaitSecs(.001);
                
        mask_time=GetSecs;
        % show response and wait for delay period
%         mask_num=ceil(rand()*4);
        while GetSecs - mask_time < show_selection+delay(trial_index), % add  250ms  so even 0 sec shows selection
%         while GetSecs  < stim_start_time+max_stim_duration, 
%         if (strcmp(resp{1,trial_index},c1_key)),
        if size(KbName(resp{1,trial_index}),2)>1
%             Screen('DrawTexture',w,tex3);    
            Screen('DrawText',w,'INVALID',xcenter-150,ycenter-180, white);
            Screen('DrawText',w,'- use proper response keys only',xcenter-400,ycenter-80, white);
            shown_corr(1,trial_index)=9;
            Screen('Flip',w);
            cat_select(1,trial_index)=0;
        elseif KbName(resp{1,trial_index})==c1_key,
        stimPtr=stimPicPtr{stim(trial_index,1)};
        Screen('DrawTexture',w,stimPtr,[],stimrect,[],[],[],[]);
%             Screen('FrameRect',w,[0 250 100],stimrect,5);
%             Screen('DrawText',w,'A',xcenter-20,ycenter-180, white);
%             Screen('FrameRect',w,[0 250 100],Lchoicerect,5);
            Screen('DrawTexture',w,choice_imagePic1,[],Lchoicerect,[],[],[],[]);
            Screen('Flip',w);
            cat_select(1,trial_index)=2;
%         elseif strcmp(resp{1,trial_index},c2_key),
        elseif KbName(resp{1,trial_index})==c2_key,
        stimPtr=stimPicPtr{stim(trial_index,1)};
        Screen('DrawTexture',w,stimPtr,[],stimrect,[],[],[],[]);
%             Screen('FrameRect',w,[0 250 100],stimrect,5);
%             Screen('DrawText',w,'B',xcenter-20,ycenter-180, white);
%             Screen('FrameRect',w,[0 250 100],Rchoicerect,5);
            Screen('DrawTexture',w,choice_imagePic2,[],Rchoicerect,[],[],[],[]);
            Screen('Flip',w);            
            cat_select(1,trial_index)=3;        
        else
%             Screen('TextSize',w,36); 
%             Screen('DrawTexture',w,tex3);    
            Screen('DrawText',w,'TOO LATE',xcenter-150,ycenter-180, white);
            shown_corr(1,trial_index)=2;
            Screen('Flip',w);
            cat_select(1,trial_index)=0;
        end;
        delay_shown(1,trial_index)=delay(trial_index);
        end;       
        rt(1,trial_index,5)=GetSecs-mask_time; % total delay time + showing answer selection
        
        if category(trial_index,1)==cat_select(1,trial_index),
            optimal_corr(1,trial_index)=1;
        else
            optimal_corr(1,trial_index)=0;
        end;

% %         mask_time=GetSecs;
% %         % present mask and wait for delay period
% %         mask_num=ceil(rand()*4);
% %         while GetSecs - mask_time < delay(trial_index),
% %         Screen('PutImage',w,mask_img{1,mask_num},[xcenter-xstimsize ycenter-ystimsize xcenter+xstimsize ycenter+ystimsize]);   
% %         Screen('FrameRect',w,black,[xcenter-xstimsize ycenter-ystimsize xcenter+xstimsize ycenter+ystimsize],2);
% %         Screen('Flip',w);           
% %         delay_shown(1,trial_index)=delay(trial_index);
% %         end;       
% %         rt(1,trial_index,5)=GetSecs-mask_time; 
% %         WaitSecs(.001);

        fb_display_time=GetSecs;
        % display feedback
        if strcmp(Verbal,'v'), % show verbal FB         
            outc_shown(1,trial_index)=outcome(trial_index);
%             Screen('TextSize',w,36);        
            %% present feedback according to outcome and response
            if ~isempty(resp{1,trial_index})  && size(KbName(resp{1,trial_index}),2)<2,                
                % create blank screen before FB
                Screen('FillRect', w, black);
                Screen('Flip', w);  
                WaitSecs(stim_off);
%                 if (strcmp(resp{1,trial_index},c1_key) & outcome(trial_index)==0) | (strcmp(resp{1,trial_index},c2_key) & outcome(trial_index)==1),
                if (KbName(resp{1,trial_index})==c1_key && outcome(trial_index)==0) || (KbName(resp{1,trial_index})==c2_key && outcome(trial_index)==1),
                Screen('DrawText',w,'CORRECT',xcenter-180,ycenter-20, [0 300 0]);        
                shown_corr(1,trial_index)=1;
                else
                Screen('DrawText',w,'INCORRECT',xcenter-210,ycenter-20, [300 0 0]);             
                shown_corr(1,trial_index)=0;
                end;      
            elseif size(KbName(resp{1,trial_index}),2)>1,
                Screen('DrawTexture',w,tex3);    
                Screen('DrawText',w,'INVALID',xcenter-150,ycenter-180, white);
                Screen('DrawText',w,'- use proper response keys only',xcenter-400,ycenter-80, white);
                shown_corr(1,trial_index)=2;
                cat_select(1,trial_index)=9;
            else
                Screen('DrawTexture',w,tex3);    
                Screen('DrawText',w,'TOO LATE',xcenter-150,ycenter-180, white);
                shown_corr(1,trial_index)=2;
            end;                                 
        else % shows scenes as FB
            outc_shown(1,trial_index)=outcome(trial_index);
            Screen('TextSize',w,36);        
            %% present feedback according to outcome and response
            if ~isempty(resp{1,trial_index})  && size(KbName(resp{1,trial_index}),2)<2,
                fb_img_index=fb_img_index+1;                
                % create blank screen before FB
                Screen('FillRect', w, black);
                Screen('Flip', w);  
                WaitSecs(stim_off);
%                 if (strcmp(resp{1,trial_index},c1_key) & outcome(trial_index)==0) || (strcmp(resp{1,trial_index},c2_key) & outcome(trial_index)==1),
                if (KbName(resp{1,trial_index})==c1_key && outcome(trial_index)==0) || (KbName(resp{1,trial_index})==c2_key && outcome(trial_index)==1),
                posfb_img_index=posfb_img_index+1;
                Screen('DrawTexture',w,tex2);    
%                 Screen('DrawTexture',w,tex3);  
                Screen('PutImage',w,fb_images{trial_index,1},FBrect);
                Screen('FrameRect',w,[0 0 0],FBrect,8);
                Screen('DrawText',w,outcome_text{1},xcenter-125,ycenter-190,[0 0 300]);
                shown_corr(1,trial_index)=1;
                trial_fb_image{1,trial_index}=FBimg{trial_index,1};
                old_posfb_image{1,posfb_img_index}=FBimg{trial_index,1};
                new_negfb_image{1,posfb_img_index}=FBimg{trial_index,2};
%                 fb_sound=hi

                else
                negfb_img_index=negfb_img_index+1;
                Screen('DrawTexture',w,tex1); 
%                 Screen('DrawTexture',w,tex3); 
                
                Screen('PutImage',w,fb_images{trial_index,2},FBrect);
                Screen('FrameRect',w,[0 0 0],FBrect,8);
                Screen('DrawText',w,outcome_text{2},xcenter-125,ycenter-190,[300 0 0]);
                shown_corr(1,trial_index)=0;
                trial_fb_image{1,trial_index}=FBimg{trial_index,2};
                old_negfb_image{1,negfb_img_index}=FBimg{trial_index,2};
                new_posfb_image{1,negfb_img_index}=FBimg{trial_index,1};

%                 fb_sound=lo
                end;      
            elseif size(KbName(resp{1,trial_index}),2)>1,
                Screen('TextSize',w,60);    
%                 Screen('DrawTexture',w,tex3);    
                Screen('DrawText',w,'INVALID',xcenter-150,ycenter-180, white);
                Screen('DrawText',w,'- use proper response keys only',xcenter-400,ycenter-80, white);
                shown_corr(1,trial_index)=2;
                cat_select(1,trial_index)=9;
            else
%                 Screen('DrawTexture',w,tex3);    
                Screen('TextSize',w,60);  
                Screen('DrawText',w,'TOO LATE',xcenter-150,ycenter-180, white);
                shown_corr(1,trial_index)=2;             
            end;  
        end;
            Screen('Flip',w);
%             sound(fb_sound);
            WaitSecs(FB);
%             while GetSecs - stim_start_time < max_stim_duration + delay(trial_index) + fb_pres_time, end; 
            rt(1,trial_index,4)=GetSecs-stim_start_time;	%total trial length?
                 
            Screen('TextSize',w,60);
            Screen('FillRect', w, black);
            Screen('DrawText',w,'+',xcenter-20,ycenter-20, white);
            Screen('Flip', w);
            WaitSecs(ISI);	
            
        % print trial info to log file
        t=trial_index;
        tmpTime=GetSecs;
        if ~isempty(resp{1,trial_index}),
            try
            fprintf(fid,'%d\t%d\t%d\t%0.3f\t%0.3f\t%0.3f\t%s\t%s\n',...
            t, stim(t),outcome(t),rt(1,trial_index,1),rt(1,trial_index,2),rt(1,trial_index,3),resp{1,trial_index},trial_fb_image{1,trial_index}); 
            catch
            fprintf(fid,'%d\t%d\t%d\t%0.3f\t%0.3f\t%0.3f\tkeymiss\n',...
            t, stim(t),outcome(t),rt(1,trial_index,1),rt(1,trial_index,2),rt(1,trial_index,3));
            end;
        else
        fprintf(fid,'%d\t%d\t%d\t%0.3f\t%0.3f\t%0.3f\tcell\n',...
        t, stim(t),outcome(t),rt(1,trial_index,1),rt(1,trial_index,2),rt(1,trial_index,3)); 
        end;  
        
        trial_blk(trial_index)=cycle;  
        
    end;  % end loop through trials	
    
    if cycle<ncycles, 
           Screen('TextSize',w,48);
            Screen('DrawText',w,'Take a break',xcenter-350,ycenter-50, white);
            Screen('Flip',w);
            WaitSecs(min_break_time);
            Screen('DrawText',w,'Take a break - tell the researcher when you',xcenter-350,ycenter-50, white);
            Screen('DrawText',w,'are ready to resume the game.',xcenter-350,ycenter, white);
            Screen('Flip',w);
            noresp=1;

        while noresp,
            [keyIsDown,secs,keyCode] = KbCheck(inputDevice(1));
            if keyIsDown && noresp,
                if strcmp(KbName(keyCode),trigger)
                noresp=0;
                end;
            end;
            WaitSecs(0.001);
        end;
    WaitSecs(0.5);  
    end;	
end; %end cycle loop

Screen('TextSize',w,48);
Screen('DrawText',w,'End of block. Thank you!',xcenter-300,ycenter);
Screen('Flip',w);

WaitSecs(2);
% clean up
Screen('CloseAll');
ShowCursor;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% save data to a file, with date/time stamp
  d=clock;
  dt=datestr(d,'ddmmmyy');
    outfile=sprintf('%s_tb_l%d_%s-%d_%s_%02.0f-%02.0f.mat',subject_code,list,FB_set,startcycle,dt,d(4),d(5));
    SM_outfile=sprintf('%s_SMlist_%d%s-%d_%s.mat',subject_code,list,FB_set,startcycle,dt);
%  % create a data structure with info about the run
   run_info.initials=subject_code;
   run_info.date=dt;
   run_info.outfile=outfile;
   run_info.problist=infilename;
   run_info.abs_start=absolute_anchor;
   if strcmp(Verbal,'v')
    try
  	save(outfile,'resp','rt','outcome','stim','stimtype','run_info','delay','outc_shown','stim_shown','stimtype_shown','delay_shown', 'shown_corr','optimal_corr', 'category','trial_blk');
    catch
	fprintf('couldn''t save %s\n saving to test.mat\n',outfile);
	save delayFB;
    end;
   else
       try
  	save(outfile,'resp','rt','outcome','stim','stimtype','run_info','trial_fb_image','old_posfb_image','old_negfb_image','new_posfb_image','new_negfb_image','delay','outc_shown','stim_shown','stimtype_shown','delay_shown', 'shown_corr','optimal_corr', 'category','trial_blk');
    save(SM_outfile,'old_posfb_image','old_negfb_image','new_posfb_image','new_negfb_image','trial_blk');   
       catch
	fprintf('couldn''t save %s\n saving to test.mat\n',outfile);
	save delayFB;
       end;		
   end;	
       
% text_file=[(1:trial_index)' stim(1:trial_index) stimtype(1:trial_index) category(1:trial_index) cat_select(1:trial_index)' optimal_corr(1:trial_index)' shown_corr(1:trial_index)'  rt(:,:,2)' delay_shown(1:trial_index)' outc_shown(1:trial_index)' ];
% onsfile=sprintf('%s_dFB_%d_%s.txt',run_info.initials,startcycle,dt);
% save(onsfile, 'text_file', '-ASCII', '-TABS');

% sum(optimal_corr)/length(find(rt(:,:,2)))
% sum(optimal_corr(1:48))/length(find(rt(:,1:48,2)))
% sum(optimal_corr(49:96))/length(find(rt(:,49:96,2)))