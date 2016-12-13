% ********* delayFBctx.m
% *************************************************************
% Component of delay FB and context manipulation - test-version
% K. Foerde, 09/27/07
% *************************************************************************
% 1. Learn S-R task with correct/incorrect FB. Delay btw stim and FB is delayed.  There  are  scanning runs of n cycles. Each cycle has n trials (2.25
%    On each trial a novel scene is shown with the FB .
% 2. Each trial consists of STIM secs of stimulus presentation, variable
% amount of delay.
%    FB is displayed for FB secs and ISI secs occurs before next trial. 
% 3. ntrials in each block and a break is given.
%**************************************************************************
clear all;

pwd
thePath.main=pwd;
[pathstr,curr_dir,ext,versn] = fileparts(pwd);
if ~strcmp(curr_dir,'dFB_adolesFMRI')
    fprintf(['You must start the experiment from the dFB_adolesFMRI directory. Go there and try again.\n']);
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
% condition = input('Is this practice (p)? ','s');
stim_set=input('Butterfly (X, Y, z, w): ', 's');
choice1=input('Flower 1: ', 's');
choice2=input('Flower 2: ', 's');
% if strcmp(condition,'p'), 
% startcycle = 1;
% else,
% image_set=input('Enter image set (a or b): ', 's');
list = input('List number: ');
% version = input('version number: ');
startcycle = input('Block number (enter 1) ');
% end;

% if strcmp(condition,'p'), 
% % infilename = sprintf('dFB_practicelist.mat');
% infilename = sprintf('tibet4_practice_o.mat');
% else,
infilename = sprintf('tibet4_probe_ord%d.mat',list);
% end;
load(infilename);
STIM=0; WARN=3; FB=2; RESP_WIN=4; stim_off=0.25; min_break_time=1;  %ISI=1; 
%STIM=1; WARN=5; FB=1.5; RESP_WIN=7; ISI=1; min_break_time=1; stim_off=0.5;
%%original timing, JYD changed for fMRI version, made timing match up with
%%Learning Phase so timing will feel the same.
%check on that STIM 0 vs 1 thing...


% define keys for each stimulus
c1_key=KbName('b');
c2_key=KbName('y');
c_key={'b' 'y'};
HideCursor;
% this next line is required to set up the psych toolbox
pixelSize = 32; 

realFMRI = input('Is this a scan session? 1 if yes, 0 if no: ');
%button_box = input('Are you using the button box? 1 if yes, 0 if no: ');

% timing of experiment
max_stim_duration=STIM+RESP_WIN;
min_stim_duration=STIM+2;
warn_time=STIM+WARN;
show_selection=1;
warning_time=1;
stim_fb_interval=0.0;
fb_pres_time=FB;
FMRI = 0; %change to 1 if you want base task to be part of loop
%trigger_wait=3;
trigger='t';

present_fb=1;
% if strcmp(condition,'p'),
% ncycles=2;
% ntrials=3;
% else,
ncycles=1;
ntrials=32; %32 not 30 to have even number of test trials for each stim (4)
% end;
trial_counter=0;

% %%%%%%%%%%%% collect this info %%%%%%%
rt=zeros(1,ncycles*ntrials,4);
resp=cell(1,ncycles*ntrials);
trial_fb_image=cell(1,ncycles*ntrials);
shown_corr=zeros(1,ncycles*ntrials);
optimal_corr=zeros(1,ncycles*ntrials);
stim_shown=zeros(1,ncycles*ntrials);
stimtype_shown=zeros(1,ncycles*ntrials);
outc_shown=zeros(1,ncycles*ntrials);
delay_shown=zeros(1,ncycles*ntrials);
cat_select=zeros(1,ncycles*ntrials);
%jitter_shown=zeros(1,ncycles*ntrials);
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

% % 
% % function [Window, Rect] = initializeScreen
% % screenNumber = 0; % 0 = main display
% % [Window, Rect] = Screen('OpenWindow',screenNumber); % open the window
% % % Set fonts
% % %Screen('TextFont',Window,'Times');
% % Screen('TextSize',Window,48);
% % Screen('FillRect', Window, 0);  % 0 = black background
% % HideCursor; % Remember to type ShowCursor later
% % 
% % 
% % 
% % %Prepares the screen
% % [Window,Rect] = initializeScreen;

xcenter=wWidth/2;
ycenter=wHeight/2;
xstimsize=120;
ystimsize=120;
fbxstimsize=200;
fbystimsize=135;
linesize=0;
yshift=50;

Lchoicerect=[xcenter-220 ycenter-260 xcenter-80 ycenter-120];
Rchoicerect=[xcenter+80 ycenter-260 xcenter+220 ycenter-120];
chosenrect=[xcenter-50 ycenter-220 xcenter+50 ycenter-120];
FBrect=[xcenter-fbxstimsize ycenter-fbystimsize xcenter+fbxstimsize ycenter+fbystimsize];
stimrect=[xcenter-xstimsize ycenter-ystimsize+yshift xcenter+xstimsize ycenter+ystimsize+yshift]
stimframerect=[xcenter-xstimsize-30 ycenter-ystimsize+yshift-30 xcenter+xstimsize+30 ycenter+ystimsize+yshift+30]



%%% create feedback graphical frames (around stimulus)
stim_image_size=[240 360];
graphic = ones(stim_image_size(1)+30,stim_image_size(2)+30,3)*255;
a_frame=graphic;
a_frame(:,:,2)=0;
a_frame(:,:,3)=0;
b_frame=graphic;
b_frame(:,:,1)=0;
b_frame(:,:,3)=0;
black_graphic = ones(stim_image_size(1),stim_image_size(2),3)*255;
black_frame=black_graphic;
black_frame(:,:,:)=0;


tex1 = Screen('MakeTexture',w,a_frame);
tex2 = Screen('MakeTexture',w,b_frame);
tex3 = Screen('MakeTexture',w,black_frame);

% load in images for weather stims (scr cell for task and base_scr for baseline_
% if strcmp(condition,'p'), 
%     stim_images=cell(1,4);
% for x=1:4, 
%     fname=sprintf('chin%d',x); 
%     stim_images(x)={imread(fname,'jpg')};
% end;
% else,
stim_images=cell(1,4);
for x=1:4, 
%     fname=sprintf('butterfly/butterfly%s%d',stim_set,x); 
%     stim_images(x)={imread(fname,'jpg')};
%     stimname=sprintf('butterfly%s%d.jpg',stim_set,x); 
%     stim_texture{x}=Screen('MakeTexture',Window,stimname)

stimPicname=sprintf('stims/butterfly%s%d',stim_set,x)
stimPic=imread(stimPicname,'jpg')
stimPicPtr{x}=Screen('MakeTexture',w,stimPic)
% buttPic{1}=imread('butterfly/butterflyz1','jpg')
% buttPicPtr=Screen('MakeTexture',w,buttPic)
% end;
end;
% 
% buttPic=imread('butterfly/butterflyz1','jpg')
% buttPicPtr{1}=Screen('MakeTexture',w,buttPic)
% buttPic=imread('butterfly/butterflyz2','jpg')
% buttPicPtr{2}=Screen('MakeTexture',w,buttPic)


fname=sprintf('stims/flower%s',choice1); 
choice_image=imread(fname,'jpg');
choice_imagePic1=Screen('MakeTexture',w,choice_image)
fname=sprintf('stims/flower%s',choice2); 
choice_image=imread(fname,'jpg');
choice_imagePic2=Screen('MakeTexture',w,choice_image)

%scr = cell(1,1); %JYD commented out
%base_scr = cell(1,1); %JYD commented out

outcome_text={'    CORRECT  ' '  INCORRECT  ' };


%%%%%%%%%%%%%%%%%%%%%%%
%% set up input devices
    numDevices=PsychHID('NumDevices');
    devices=PsychHID('Devices');
    if realFMRI==1,
        for n=1:numDevices,
            if (findstr(devices(n).transport,'USB') & findstr(devices(n).usageName,'Keyboard') & (devices(n).productID==16385 | devices(n).vendorID==6171 | devices(n).totalElements==274)),
                inputDevice=n;
            %else,
            %    inputDevice=6; % my keyboard
            end;
        end;
        fprintf('Using Device #%d (%s)\n',inputDevice,devices(inputDevice).product);
    else,
        for n=1:numDevices,
            if (findstr(devices(n).transport,'USB') & findstr(devices(n).usageName,'Keyboard')),
                inputDevice=[n n];
                break,
            elseif (findstr(devices(n).transport,'Bluetooth') & findstr(devices(n).usageName,'Keyboard')),
                inputDevice=[n n];
                break,
            elseif findstr(devices(n).transport,'ADB') & findstr(devices(n).usageName,'Keyboard'),
                inputDevice=[n n];
            end;
        end;
        fprintf('Using Device #%d (%s)\n',inputDevice,devices(n).product);
    end;

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% print instructions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Screen('TextSize',w,24);
%screen('CopyWindow',scr{16},w);

 %if strcmp(condition,'p'), %was commented out, JYD put back in with other
 if strcmp(realFMRI,'1'), 
	inst={{'Now you will continue making flower selections,'} ...
	{'but you will not be told whether you are CORRECT or INCORRECT.'} ...
    {''}...
    {'It is still important that you give an answer on each trial.'}...
    {''}...
    {'Press the "s" key to begin'}};
    for x=1:size(inst,2),
        Screen('DrawText',w,inst{x}{:},100,100+x*35, white);
    end;
    Screen('Flip',w);

     
    noresp=1;
    while noresp,
            [keyIsDown,secs,keyCode] = KbCheck(inputDevice(2));
            if keyIsDown & noresp,
                noresp=0;
            end;
            WaitSecs(0.001);
    end;
% end;
  else,
%show this until trials starts
 Screen('TextSize',w,64);
 Screen('FillRect', w, black);
 Screen('DrawText',w,'Get Ready!', xcenter-220, ycenter-20, white);
 Screen('Flip', w);
     noresp=1;
     if realFMRI==1,
     while noresp,
             [keyIsDown,secs,keyCode] = KbCheck(inputDevice(1));%JYD needs to be 1 to collect responses from buttonbox
             if keyIsDown & noresp,
                 if strcmp(KbName(keyCode),trigger)
                 noresp=0;
                 end;
             end;
             WaitSecs(0.001);
     end;
     else,
     while noresp,
             [keyIsDown,secs,keyCode] = KbCheck(inputDevice(2));
             if keyIsDown & noresp,
                 if strcmp(KbName(keyCode),trigger)
                 noresp=0;
                 end;
             end;
             WaitSecs(0.001);
     end;
     end;
 end;
WaitSecs(0.5);  % prevent key spillover    
%DisableKeysForKbCheck(KbName('t')); % So trigger is no longer detected
%%JYDXX in case we do need this    

dd=clock;
fprintf(fid,'Started: %s %2.0f:%02.0f:%02.0f\n',date,dd(4),dd(5),dd(6));
absolute_anchor=GetSecs; % get absolute starting time anchor
noresp=0;  % set to zero initially

% loop through trials
for cycle=startcycle:ncycles,
%%%%JYD just took out to see about timing 11/14/12%%%%    
%noresp=1;
%Screen('TextSize',w,64);
%Screen('FillRect', w, black);
%Screen('DrawText',w,'GET READY (t)',xcenter-220,ycenter-20, white);
%Screen('Flip', w);
%    while noresp,
%            [keyIsDown,secs,keyCode] = KbCheck(inputDevice(1));
%            if keyIsDown & noresp,
%                if strcmp(KbName(keyCode),trigger)
%                noresp=0;
%                end;
%            end;
%            WaitSecs(0.001);
%    end;
%            WaitSecs(0.001);
%    Screen('TextSize',w,64);
%    Screen('FillRect', w, black);
%    Screen('DrawText',w,'+',xcenter-20,ycenter-20, white);
 %   Screen('Flip', w);  
%%%%JYD just took out to see about timing 11/14/12%%%%
    anchor=GetSecs; % get absolute starting time anchor
    %WaitSecs(trigger_wait); % start manual so wait for first 4 measurements to be discarded
%JYD do we need this? Ask Elizabeth

    for trial=1:ntrials, 
%             while GetSecs - start_time < max_stim_duration + delay(trial_index) + fb_pres_time, end; 	
            
		trial_index=(cycle-1)*ntrials+trial;
         while GetSecs - anchor < trialtimes(trial_index,1), end; %waits to synch beginning of trial with 'true' start %JYD commented-back-in for fMRI

        %present image + questions
%         Screen('PutImage',w,stim_images{stim(trial_index,1)},stimrect);   
        stimPtr=stimPicPtr{stim(trial_index,1)};
        Screen('DrawTexture',w,stimPtr,[],stimrect,[],[],[],[]);
            
            
%          Screen('FrameRect',w,[0 250 100],stimrect,5);
%         Screen('FrameRect',w,[0 250 100],Lchoicerect,5);
%         Screen('FrameRect',w,[0 250 100],Rchoicerect,5);
%         Screen('PutImage',w,choice_image{1},Lchoicerect);
%         Screen('PutImage',w,choice_image{2},Rchoicerect);
        Screen('DrawTexture',w,choice_imagePic1,[],Lchoicerect,[],[],[],[]);
        Screen('DrawTexture',w,choice_imagePic2,[],Rchoicerect,[],[],[],[]);
%         Screen('FrameRect',w,black,[xcenter-xstimsize ycenter-ystimsize xcenter+xstimsize ycenter+ystimsize],2);
        Screen('DrawText',w,'     ?',xcenter-112,ycenter-250, white);
        Screen('Flip',w);
        stim_start_time=GetSecs;
		rt(1,trial_index,1)=stim_start_time-anchor;
   		rt(1,trial_index,3)=stim_start_time-absolute_anchor;     
        stim_shown(1,trial_index)=stim(trial_index);
        stimtype_shown(1,trial_index)=stimtype(trial_index);
%         rt(1,trial_index,5)=GetSecs-rt(1,trial_index,5);
%         WaitSecs(max_stim_duration);
%         %ask for response
% 		Screen('PutImage',w,stim_images{stim(trial_index,1)},[xcenter-xstimsize ycenter-ystimsize xcenter+xstimsize ycenter+ystimsize]);
%         Screen('FrameRect',w,black,[xcenter-xstimsize ycenter-ystimsize xcenter+xstimsize ycenter+ystimsize],2);
%         Screen('DrawText',w,'A or B ?',xcenter-112,ycenter-180, white);
%         Screen('Flip',w);        
%         resp_start_time=GetSecs;        
% 		rt(1,trial_index,3)=resp_start_time-anchor;        

        % to collect responses and RTs in OSX
        noresp=1;
%         warned=0;
        if WARN==0,        warned=1;else, warned=0; end;
        
%JYD-next line is different than in the learning script, why?
        while (GetSecs - stim_start_time < max_stim_duration) && noresp,   
        [keyIsDown,secs,keyCode] = KbCheck(inputDevice(1));%JYD needs to be 1 to collect responses from buttonbox
            if keyIsDown && noresp && (strcmp(KbName(keyCode),trigger)~=1), %JYD commented back in for fmri
                tmp=KbName(keyCode);
%                 if sum(strcmp({tmp(1)},c_key)>0),
%                 resp(1,trial_index)={tmp(1)};
                if(keyCode(c1_key) | keyCode(c2_key)),
                resp(1,trial_index)={tmp};
                rt(1,trial_index,2)=GetSecs-stim_start_time; 
                noresp=0;
                end;
            WaitSecs(.001);
            end;	  
               
            if  ~warned & (GetSecs - stim_start_time > warn_time)        
%         Screen('PutImage',w,stim_images{stim(trial_index,1)},stimrect);
        Screen('DrawTexture',w,stimPtr,[],stimrect,[],[],[],[]);
%         Screen('FrameRect',w,black,[xcenter-xstimsize ycenter-ystimsize xcenter+xstimsize ycenter+ystimsize],2);
        Screen('DrawText',w,'PLEASE RESPOND',xcenter-262,ycenter-370, white);
%         Screen('FrameRect',w,[0 250 100],stimrect,5);
%         Screen('FrameRect',w,[0 250 100],Lchoicerect,5);
%         Screen('FrameRect',w,[0 250 100],Rchoicerect,5);
%         Screen('PutImage',w,choice_image{1},Lchoicerect);
%         Screen('PutImage',w,choice_image{2},Rchoicerect);
                Screen('DrawTexture',w,choice_imagePic1,[],Lchoicerect,[],[],[],[]);
        Screen('DrawTexture',w,choice_imagePic2,[],Rchoicerect,[],[],[],[]);
        Screen('DrawText',w,'     ?',xcenter-112,ycenter-250, white);       
        Screen('Flip',w);     
           warned=1;
           end;                        
        WaitSecs(0.001);            
        end;
        
%JYD grabbed from learning phase script to add delay to probe
        mask_time=GetSecs;
        % show response and wait for delay period
        while GetSecs - mask_time < show_selection+delay(trial_index), % add  250ms  so even 0 sec shows selection
        if size(KbName(resp{1,trial_index}),2)>1
%             Screen('DrawTexture',w,tex3);    
            Screen('DrawText',w,'INVALID',xcenter-150,ycenter-180, white);
            Screen('DrawText',w,'- use proper response keys only',xcenter-400,ycenter-80, white);
            shown_corr(1,trial_index)=9;
            Screen('Flip',w);
        %JYDelse
%         Screen('PutImage',w,stim_images{stim(trial_index,1)},stimrect);
        %JYDScreen('DrawTexture',w,stimPtr,[],stimrect,[],[],[],[]);
%         Screen('FrameRect',w,[0 250 100],stimrect,5);
%                 Screen('DrawText',w,'     ?',xcenter-112,ycenter-250, black); 
        %         Screen('FrameRect',w,black,[xcenter-xstimsize ycenter-ystimsize xcenter+xstimsize ycenter+ystimsize],2);
        %JYDScreen('Flip',w);
        elseif KbName(resp{1,trial_index})==c1_key,
                stimPtr=stimPicPtr{stim(trial_index,1)};
                Screen('DrawTexture',w,stimPtr,[],stimrect,[],[],[],[]);
                Screen('DrawTexture',w,choice_imagePic1,[],Lchoicerect,[],[],[],[]);
                Screen('Flip',w);
                %cat_select(1,trial_index)=2;
                %         elseif strcmp(resp{1,trial_index},c2_key),
            elseif KbName(resp{1,trial_index})==c2_key,
                stimPtr=stimPicPtr{stim(trial_index,1)};
                Screen('DrawTexture',w,stimPtr,[],stimrect,[],[],[],[]);
                Screen('DrawTexture',w,choice_imagePic2,[],Rchoicerect,[],[],[],[]);
                Screen('Flip',w);
                %cat_select(1,trial_index)=3;
                else
                Screen('DrawText',w,'TOO LATE',xcenter-150,ycenter-180, white);
                shown_corr(1,trial_index)=2;
                Screen('Flip',w);
                cat_select(1,trial_index)=0;
        end;
        delay_shown(1,trial_index)=delay(trial_index);
        end;
        %WaitSecs(1); %JYD commented out because I think the delay takes
        %the place of this
%         while (GetSecs - stim_start_time < RESP_WIN),
                    Screen('FillRect', w, black);
        Screen('Flip', w);  
                WaitSecs(.25); %JYD what is this? Is it needed, should it be removed?
%         end;
      
        if category(trial_index,1)==cat_select(1,trial_index),
            optimal_corr(1,trial_index)=1;
        else,
            optimal_corr(1,trial_index)=0;
        end;

        %JYD commented the line below back in in hopes of fixing timing
        %"end" on line 498
         %if (GetSecs - anchor < trialtimes(trial_counter+1)), 
        Screen('TextSize',w,64);
        Screen('FillRect', w, black);
        Screen('DrawText',w,'+',xcenter-20,ycenter-20, white);
        Screen('Flip', w);
         %end;
		rt(1,trial_index,4)=GetSecs-stim_start_time;
             %WaitSecs(ISI+jitter(trial_index)); %JYD added jitter
             
        % print trial info to log file
        t=trial_index;
        tmpTime=GetSecs;
        if ~isempty(resp{1,trial_index}),
            try
            fprintf(fid,'%d\t%d\t%d\t%0.3f\t%0.3f\t%0.3f\t%s\n',...
            t, stim(t),outcome(t),rt(1,trial_index,1),rt(1,trial_index,2),rt(1,trial_index,3),resp{1,trial_index}); 
            catch,
            fprintf(fid,'%d\t%d\t%d\t%0.3f\t%0.3f\t%0.3f\tkeymiss\n',...
            t, stim(t),outcome(t),rt(1,trial_index,1),rt(1,trial_index,2),rt(1,trial_index,3));
            end;
        else,
        fprintf(fid,'%d\t%d\t%d\t%0.3f\t%0.3f\t%0.3f\tcell\n',...
        t, stim(t),outcome(t),rt(1,trial_index,1),rt(1,trial_index,2),rt(1,trial_index,3)); 
        end;  
  
    end;  % end loop through trials	
    
    if cycle<ncycles, 
           Screen('TextSize',w,44);
            Screen('DrawText',w,'Take a break',xcenter-350,ycenter-50, white);
            Screen('Flip',w);
            WaitSecs(min_break_time);
            Screen('DrawText',w,'Take a break - press "t" when you are ready to resume the game.',xcenter-350,ycenter-50, white);
            Screen('Flip',w);
            noresp=1;
        while noresp,
            [keyIsDown,secs,keyCode] = KbCheck(inputDevice(1));
            if keyIsDown & noresp,
                if strcmp(KbName(keyCode),trigger)
                noresp=0;
                end;
            end;
            WaitSecs(0.001);
        end;
    WaitSecs(0.5);  
    end;	
end; %end cycle loop
%Screen('TextSize',w,44);
Screen('TextSize',w,36);
Screen('DrawText',w,'End of the game. Thank you! ',xcenter-300,ycenter-100);
%Screen('DrawText',w,'Please take a break. Tell the researcher',xcenter-350,ycenter-50, white);
%Screen('DrawText',w,'when you are ready to resume the game.',xcenter-350,ycenter, white);
WaitSecs(2);
Screen('Flip',w);

WaitSecs(4);
% clean up
Screen('CloseAll');
ShowCursor;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% save data to a file, with date/time stamp
  d=clock;
  dt=datestr(d,'ddmmmyy');
    outfile=sprintf('%s_tb_prb_%d_%s_%02.0f-%02.0f.mat',subject_code,list,dt,d(4),d(5));

%  % create a data structure with info about the run
   run_info.initials=subject_code;
   run_info.date=date;
   run_info.outfile=outfile;
%    run_info.condition=condition;
   run_info.problist=infilename;
   run_info.abs_start=absolute_anchor;
%    run_info.trainlength=trainlength;
%    run_info.stim_set1=stim_set1;
%    run_info.base_img=base_img;
   try,
  	save(outfile,'resp','rt','outcome','stim','stimtype','run_info','delay','outc_shown','stim_shown','stimtype_shown','delay_shown', 'shown_corr','optimal_corr', 'category');
   catch,
	fprintf('couldn''t save %s\n saving to test.mat\n',outfile);
	save delayFB;
   end;		
   
text_file=[(1:trial_index)' stim(1:trial_index) stimtype(1:trial_index) category(1:trial_index) cat_select(1:trial_index)' optimal_corr(1:trial_index)' shown_corr(1:trial_index)'  rt(:,:,2)' delay_shown(1:trial_index)' outc_shown(1:trial_index)' ];
onsfile=sprintf('%s_dFBprb_%s.txt',run_info.initials,dt)
save(onsfile, 'text_file', '-ASCII', '-TABS')