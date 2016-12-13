% ********* delayFB_SMtest.m
% *************************************************************
% Component of delay FB and context manipulation - Subsequent memory test
% K. Foerde, 01/14/08
% *************************************************************************
% 1. OLD and NEW scenes.
% 2. Determine OLD / NEW 
% 3. Determine confidence for old item.
% In Learning Task outdorr scenes= correct and indoor = INCORRECT
% in SM test equal number old/new outdoor and old/new indoor scenes are
% shown, but indoor ~= outdoor number
%**************************************************************************

% make list for targets that balances early late and delay type images
% throughout test session. Have foil list that balances image type
% througout list

clear all;

rand('state',sum(100*clock));  % seed the random number generator
subject_code = input('Enter subject code: ', 's');
FB_set=input('Enter FB set (i, o, alli, allo, allmix, svlo): ', 's');
% task_set=input('Enter task set (1 or 2; 0 for alli and allo): ');
% image_set=input('Enter image set (A or B): ', 's');
% if strcmp(image_set,'A'),
%     foil_set='B';
% elseif strcmp(image_set,'B'),
%     foil_set='A';
% end;
list = input('List number: ');
startcycle=1;
endcycle = input('End Block number (enter 4) ');

%% load in list:
dt=datestr(clock,'ddmmmyy');

% infilename = sprintf('%s_SMlist_%s%d%d%s_%s.mat',subject_code,image_set,list,startcycle,FB_set,dt);
infilename = sprintf('%s_SMlist_%d%s-%d_%s.mat',subject_code,list,FB_set,startcycle,dt);
load(infilename); % loads in learning file to get OLD/NEW images

num_blks=4;
STIM=1; FB=2.5; ISI=0.5; min_break_time=1;


% define keys for each stimulus
c_key={'7' '9'};
conf_key={'1' '2' '3' '4'};
%conf_key={'1' '2' '3' '4' '5' '6'};
% c1_key={'1'};
% c2_key={'2'};
% c3_key={'3'};
% c4_key={'4'};
oldnew_keys={'n' 'm'};
frame_key={'r' 'b'};

HideCursor;
% this next line is required to set up the psych toolbox
pixelSize = 32; 

realFMRI = input('Is this a scan session? 1 if yes, 0 if no: ');
% button_box = input('Are you using the button box? 1 if yes, 0 if no: ');

% timing of experiment
stim_duration=STIM;
max_stim_duration=STIM;
warning_time=1;
stim_fb_interval=0.0;
fb_pres_time=FB;
% FMRI = 0; %change to 1 if you want base task to be part of loop
% trigger_wait=6;
trigger='t';
present_fb=1;
trial_counter=0;


%% OSX %%
%set up screens
screens=Screen('Screens');
screenNumber=max(screens);
w=Screen('OpenWindow', screenNumber,0,[],32,2);
[wWidth, wHeight]=Screen('WindowSize', w);
black=BlackIndex(w); % Should equal 0.
white=WhiteIndex(w); % Should equal 255.
grayLevel=120;
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
xstimsize=135;%180;
ystimsize=135;%120;
linesize=0;

%%% create feedback graphical frames (around stimulus)
stim_image_size=[240 360];
graphic = ones(stim_image_size(1)+30,stim_image_size(2)+30,3)*255;
a_frame=graphic;
a_frame(:,:,2)=0;
a_frame(:,:,3)=0;
b_frame=graphic;
b_frame(:,:,1)=0;
b_frame(:,:,3)=0;
% bl_stim_image_size=[170 355];
% bl_graphic = ones(bl_stim_image_size(1)+20,bl_stim_image_size(2)+20,3)*255;
% black_frame=bl_graphic;
% black_frame(:,:,:)=0;

tex1 = Screen('MakeTexture',w,a_frame);
tex2 = Screen('MakeTexture',w,b_frame);
% tex3 = Screen('MakeTexture',w,black_frame);



%determine how many OLD images and then double the number 
num_SMtrials=length(old_posfb_image)+length(old_negfb_image);
%determine at which trial to take break
nts=floor(2*num_SMtrials/num_blks);
brk=[nts:nts:2*num_SMtrials]+1;

%determine order of OLD and NEW images (random)
% make sure each test block has equal old new number
load old_new_list; % 50/50 old/new
%old_new_list=old_new_list(randperm(2*num_SMtrials)); % this does not
%distrbute evenly among test blocks - below does.

% this ensure that blocks have an even number (so old==new)
if rem((brk(1)-1),2)==1,
   nts=nts+1
   brk=[nts:nts:2*num_SMtrials]+1;
end;

for nb=1,
    old_new_list(1:brk(nb)-1)=old_new_list(randperm(length(1:brk(nb)-1)));
end;
for nb=2:num_blks-1,
    old_new_list(brk(nb-1):brk(nb)-1)=old_new_list(randperm(length(brk(nb-1):brk(nb)-1))+brk(nb)-1);
end;
for nb=num_blks,
    old_new_list(brk(nb-1):2*num_SMtrials)=old_new_list(randperm(length(brk(nb-1):2*num_SMtrials))+brk(nb-1));
end;
% counterbalance early and late images early and late in test
% make block_list
% for b=1:max(trial_blk),
%     block_list(b:max(trial_blk):length(trial_blk))=b;
% end;
for b=1:endcycle,
    block_list(b:endcycle:length(trial_blk))=b;
end;
% load block_list;
% take image from each train_block and test in each test_block
%do this separate for pos/neg
[sorted,sortidx]=sort(block_list(1:length(old_posfb_image)));
old_posfb_image=old_posfb_image(sortidx);
[sorted,sortidx]=sort(block_list(1:length(old_negfb_image)));
old_negfb_image=old_negfb_image(sortidx);

% make indoor/outdoor list (proportion determined by pos/neg in training)
pos_neg_list=[ones(1,length(old_posfb_image)) 2*ones(1,length(old_negfb_image))];
% make a separate random order for new and old
old_pos_neg_list=pos_neg_list(randperm(length(pos_neg_list)));
new_pos_neg_list=pos_neg_list(randperm(length(pos_neg_list)));

% add more images to the end of the new list to equal the old list
more_images_file=sprintf('more_images_%s.mat',FB_set)
load(more_images_file); % need to fix more_images to have separate indoor/outdoor
% % the more_images file variables have been renamed now
% % if strcmp(FB_set,'i'),
% %     more_pos=more_ind;
% %     more_neg=more_out;
% % elseif strcmp(FB_set,'o'),
% %     more_pos=more_out;
% %     more_neg=more_ind;
% % end;
new_pos_images1=[new_posfb_image'; more_pos]; % should be list B if used list A to train
new_neg_images1=[new_negfb_image'; more_neg]; % need more new_neg_images

if length(new_neg_images1) > length(new_pos_images1)
new_neg_images=   new_neg_images1(1:length(old_negfb_image))
new_pos_images=[new_pos_images1; new_neg_images1(length(old_negfb_image)+1:length(old_posfb_image))];
elseif length(new_pos_images1) > length(new_neg_images1) 
new_pos_images=   new_pos_images1(1:length(old_posfb_image))
new_neg_images=[new_neg_images1; new_pos_images1(length(old_posfb_image)+1:length(old_negfb_image))];
end    

ncycles=1;
ntrials=2*num_SMtrials;
% %%%%%%%%%%%% collect this info %%%%%%%
%JYD check that nothing was added to RT info that gets saved
rt=zeros(ncycles*ntrials,1,4);
resp=cell(ncycles*ntrials,1,3);
trial_fb_image=cell(ncycles*ntrials,1);
% correct_w=zeros(1,ncycles*ntrials*nxcycles);
% correct_win=zeros(1,ncycles*ntrials*nxcycles);
% resp_w=zeros(1,ncycles*ntrials*nxcycles);
% rt_w=zeros(1,ncycles*ntrials*nxcycles);
% nfiftypc_w=zeros(1,ncycles*ntrials*nxcycles);

% make list of images threading old and new fb images together 
old_pos_trial=1;
old_neg_trial=1;
new_pos_trial=1;
new_neg_trial=1;
old=1;
new=1;

for x=1:2*num_SMtrials,
    if old_new_list(x)==1,
        if old_pos_neg_list(old)==1,            
        test_image{x}=old_posfb_image{old_pos_trial};
        old_pos_trial=old_pos_trial+1;
        elseif old_pos_neg_list(old)==2,
        test_image{x}=old_negfb_image{old_neg_trial};
        old_neg_trial=old_neg_trial+1;
        end;
        old=old+1;
    elseif old_new_list(x)==2,
        if new_pos_neg_list(new)==1,     
        test_image{x}=new_pos_images{new_pos_trial};
        new_pos_trial=new_pos_trial+1;
        elseif new_pos_neg_list(new)==2,     
        test_image{x}=new_neg_images{new_neg_trial};
        new_neg_trial=new_neg_trial+1;
        end;
        new=new+1;
    end;
end;


d=clock;
startload=sprintf('%02.0f-%02.0f-%02.0f',d(4),d(5),d(6));
% %now read in all the test images from list aboeve
% fb_images=cell(1,length(test_image));
% for x=1:length(test_image),
%     FBimg=test_image{x};
%          fname=sprintf('FB_images_4PD/outdoor/grp%d%so/%s',task_set,image_set,FBimg); 
%         try,
%          fb_images(x)={imread(fname)};
%         catch
%             try 
%          fname=sprintf('FB_images_4PD/outdoor/grp%d%so/%s',task_set,foil_set,FBimg); 
%          fb_images(x)={imread(fname)};
%             catch
%                  try 
%          fname=sprintf('FB_images_4PD/indoor/grp%d%si/%s',task_set,image_set,FBimg); 
%          fb_images(x)={imread(fname)};
%                  catch
%          fname=sprintf('FB_images_4PD/indoor/grp%d%si/%s',task_set,foil_set,FBimg); 
%          fb_images(x)={imread(fname)};               
%                 end;
%             end;
%         end;
% end;

% now read in all the test images from list aboeve
fb_images=cell(1,length(test_image));
for x=1:length(test_image),
    FBimg=test_image{x};
         fname=sprintf('stims/%s',FBimg); 
         fb_images(x)={imread(fname)};
end;

% %now read in all the test images from list aboeve
% fb_images=cell(1,length(test_image));
% for x=1:length(test_image),
%     FBimg=test_image{x};
%     fname=sprintf('FB_images/outdoor/%s',FBimg); 
%     try,
%         fb_images(x)={imread(fname)};
%     catch
%         fname=sprintf('FB_images/indoor/%s',FBimg); 
%         fb_images(x)={imread(fname)};
%     end;
% end;

d=clock;
endload=sprintf('%02.0f-%02.0f-%02.0f',d(4),d(5),d(6));


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
else,
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
Screen('TextSize',w,28);
%screen('CopyWindow',scr{16},w);

% if strcmp(condition,'p'), 

% 	{'You will also see an equal number of new images.'} ...
%     {'OLD and NEW images will be intermixed.'} ...
% 	{'On each trial you will be presented with an image. '} ...
% 	{'First you will decide whether the image is OLD or NEW.'}...
%         {'Press (7) for an OLD image and (9) for a NEW image.'}...
%     {'Then you will indicate how sure you are about that decision.'} ...
% 	{'You will determine whether you are'}...
%     {'Completely certain (1), Very sure (2), Pretty sure (3) or Guessing (4)'}...
%     {''}...

	inst={{'Now you will see some pictures of objects and animals. You will have'}... 
    {'seen some of these pictures from the part of the game you just played inside the scanner.'}...
    {'On each trial you will see a picture. Your job is to remember if you saw this picture while'}...
    {'you were inside the scanner.'}...
    {' '}...
    {'An image is OLD if you saw it during the game you played while inside the scanner.'}...
    {'An image is NEW if you DID NOT see it during the game you played while inside the scanner.'}...
    {'Press (7) for an OLD image and (9) for a NEW image. These instructions will be on the'}... 
    {'screen for each trial.'}...
    {' '}...
    {'Then you will be asked how sure you are about that decision.'}...
    {'You will press a button to tell us if you are '}...
    {'(1) Completely certain, (2) Very sure, (3) Pretty sure, or (4) Guessing'}...
    {' '}...
    {'We will do the first few trials together!'}...
    {'Press any key to begin'}};
    %for x=1:size(inst,2),
    %Screen('DrawText',w,inst{x}{:},100,100+x*55, white);
    %end;
    for x=1:size(inst,2),
        Screen('DrawText',w,inst{x}{:},100,100+x*35, white);
    end;
    Screen('Flip',w);

    noresp=1;
    while noresp,
            [keyIsDown,secs,keyCode] = KbCheck(inputDevice(1));
            if keyIsDown & noresp,
                noresp=0;
            end;
            WaitSecs(0.001);
    end;
    WaitSecs(0.5);  % prevent key spillover    
     
%show this until trials starts
Screen('TextSize',w,64);
% Screen('FillRect', w, grayLevel);
Screen('DrawText',w,'+',xcenter-20,ycenter-20, white);
Screen('Flip', w);
 
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
% WaitSecs(trigger_wait); % start manual so wait for first 4 measurements
% to be discarded
dd=clock;
fprintf(fid,'Started: %s %2.0f:%02.0f:%02.0f\n',date,dd(4),dd(5),dd(6));
anchor=GetSecs; % get absolute starting time anchor
noresp=0;  % set to zero initially

% loop through trials	
for cycle=1:ncycles,
	for trial=1:ntrials, 

        
    Screen('TextSize',w,64);
%     Screen('FillRect', w, grayLevel);
    Screen('Flip', w);
    WaitSecs(ISI);		
            
    trial_index=(cycle-1)*ntrials+trial;
        if sum(trial_index==brk)==1,
            Screen('TextSize',w,44);
            Screen('DrawText',w,'Take a short break',xcenter-200,ycenter-50, white);
            Screen('Flip',w);
            WaitSecs(min_break_time);
            Screen('DrawText',w,'Take a short break',xcenter-200,ycenter-50, white);
            Screen('DrawText',w,'- press any key to continue',xcenter-300,ycenter+150, white);
            Screen('Flip',w);
            noresp=1;
        while noresp,
        [keyIsDown,secs,keyCode] = KbCheck(inputDevice(1));
            if keyIsDown & noresp,
            noresp=0;
            end;
        WaitSecs(0.001);
        end;
            WaitSecs(ISI);	
        end;
    
		stim_start_time=GetSecs;
   
        Screen('TextSize',w,36);
        Screen('PutImage',w,fb_images{trial_index},[xcenter-xstimsize ycenter-ystimsize xcenter+xstimsize ycenter+ystimsize]);
        Screen('DrawText',w,'OLD        or      NEW ?',xcenter-175,ycenter-220, white);
        Screen('TextSize',w,24);
        Screen('DrawText',w,'(7)',xcenter-150,ycenter-180, white);
        Screen('DrawText',w,'(9) ',xcenter+100,ycenter-180, white);
        Screen('FrameRect',w,black,[xcenter-xstimsize+linesize ycenter-ystimsize+linesize xcenter+xstimsize-linesize ycenter+ystimsize-linesize],8);
        Screen('Flip',w);
        trial_fb_image{trial_index,1}=fb_images{trial_index};
		start_time=GetSecs;
		rt(trial_index,1,1)=start_time-anchor;

        % to collect responses and RTs in OSX
        noresp=1;
        warned=0;

        while noresp,   
        [keyIsDown,secs,keyCode] = KbCheck(inputDevice(2));
           if keyIsDown & noresp, 
%             if keyIsDown & noresp & ((strcmp(KbName(keyCode),c1_key)==1) | (strcmp(KbName(keyCode),c2_key)==1)),               
                tmp=KbName(keyCode);
                if sum(strcmp({tmp(1)},c_key)>0),
                resp(trial_index,1,1)={tmp(1)};
                rt(trial_index,1,2)=GetSecs-start_time; 
                noresp=0;
                end;
            end;	
            WaitSecs(.001);
        end;
        WaitSecs(0.25);  % prevent key spillover    
        noresp=1;
 
%         % non-R-K version
%         Screen('TextSize',w,36);
%         Screen('DrawText',w,'Completely certain          Very sure         Pretty sure      Just Guessing',xcenter-525,ycenter-220, white);
%         Screen('TextSize',w,20);
%         Screen('DrawText',w,'        1                                                         2                                             3                                                  4' ,xcenter-425,ycenter-160, white);
%         Screen('Flip',w);    
        % 6 item confidence scale
        Screen('TextSize',w,36);
%         Screen('DrawText',w,'Absolutely certain ',xcenter-325,ycenter-300, white);
%         Screen('DrawText',w,'Certain            ',xcenter-325,ycenter-220, white);
        Screen('DrawText',w,'Completely Certain            ',xcenter-325,ycenter-220, white);
        Screen('DrawText',w,'Very sure          ',xcenter-325,ycenter-140, white);
        Screen('DrawText',w,'Pretty sure        ',xcenter-325,ycenter-60, white);
%         Screen('DrawText',w,'Uncertain          ',xcenter-325,ycenter+20, white);
        Screen('DrawText',w,'Just Guessing      ',xcenter-325,ycenter+100, white);
        Screen('TextSize',w,30);
%         Screen('DrawText',w,'(1)',xcenter+125,ycenter-300, white);
%         Screen('DrawText',w,'(2)',xcenter+125,ycenter-220, white);
%         Screen('DrawText',w,'(3)',xcenter+125,ycenter-140, white);
%         Screen('DrawText',w,'(4)',xcenter+125,ycenter-60, white);
%         Screen('DrawText',w,'(5)',xcenter+125,ycenter+20, white);
%         Screen('DrawText',w,'(6)',xcenter+125,ycenter+100, white);

        Screen('DrawText',w,'(1)',xcenter+125,ycenter-220, white);
        Screen('DrawText',w,'(2)',xcenter+125,ycenter-140, white);
        Screen('DrawText',w,'(3)',xcenter+125,ycenter-60, white);
%         Screen('DrawText',w,'(5)',xcenter+125,ycenter+20, white);
        Screen('DrawText',w,'(4)',xcenter+125,ycenter+100, white);
        Screen('Flip',w); 
        
            while noresp==1,
            [keyIsDown,secs,keyCode] = KbCheck(inputDevice(2));
%                 if keyIsDown & noresp ((strcmp(KbName(keyCode),c1_key)==1) | (strcmp(KbName(keyCode),c2_key)==1) | (strcmp(KbName(keyCode),c3_key)==1)  | (strcmp(KbName(keyCode),c4_key)==1)),
                if keyIsDown & noresp,% & (sum(strcmp(KbName(keyCode),c_key)>0)), % | (strcmp(KbName(keyCode),c2_key)==1) | (strcmp(KbName(keyCode),c3_key)==1)  | (strcmp(KbName(keyCode),c4_key)==1)),
                    tmp=KbName(keyCode);
                    if sum(strcmp({tmp(1)},conf_key)>0),
                    resp(trial_index,1,2)={tmp(1)};
                    rt(trial_index,1,3)=GetSecs-start_time; 
                    noresp=0;
                    end;
                end;
            WaitSecs(.001);
            end;
            WaitSecs(0.25);  % prevent key spillover       
            noresp=1;
            
% %             if strcmp(FB_set,'allo') | strcmp(FB_set,'alli'),               
% %             Screen('TextSize',w,36);
% %             Screen('PutImage',w,fb_images{trial_index},[xcenter-xstimsize ycenter-ystimsize xcenter+xstimsize ycenter+ystimsize]);
% %             Screen('DrawText',w,'What color was the picture frame?',xcenter-275,ycenter-260, white);
% %             Screen('DrawText',w,'RED        or      BLUE ?',xcenter-175,ycenter-220, white);
% %             Screen('TextSize',w,24);
% %             Screen('DrawText',w,'(r)',xcenter-150,ycenter-180, white);
% %             Screen('DrawText',w,'(b) ',xcenter+100,ycenter-180, white);
% %             Screen('FrameRect',w,black,[xcenter-xstimsize+linesize ycenter-ystimsize+linesize xcenter+xstimsize-linesize ycenter+ystimsize-linesize],8);
% %             Screen('Flip',w);
% %             trial_fb_image{trial_index,1}=fb_images{trial_index};
% %             start_time=GetSecs;
% % 
% %                 while noresp==1,   
% %                 [keyIsDown,secs,keyCode] = KbCheck(inputDevice(2));
% %                    if keyIsDown & noresp, 
% %         %             if keyIsDown & noresp & ((strcmp(KbName(keyCode),c1_key)==1) | (strcmp(KbName(keyCode),c2_key)==1)),               
% %                         tmp=KbName(keyCode);
% %                         if sum(strcmp({tmp(1)},frame_key)>0),
% %                         resp(trial_index,1,3)={tmp(1)};
% %                         rt(trial_index,1,4)=GetSecs-start_time; 
% %                         noresp=0;
% %                         end;
% %                     end;	
% %                     WaitSecs(.001);
% %                 end;
% %                 WaitSecs(0.25);  % prevent key spillover    
% %                 noresp=1;
% %             end;
        
%         % print trial info to log file
        t=trial_index;
        tmpTime=GetSecs;
        if ~isempty(resp{t,1}),
            try
            fprintf(fid,'%d\t%s\t%d\t%s\t%s\t%s\t%0.3f\t%0.3f\t%0.3f\n',...
            t,test_image{t},old_new_list(t),resp{t,1},resp{t,2},resp{t,3},rt(t,1),rt(t,2),rt(t,3)); 
            catch,
            fprintf(fid,'%d\t%s\t%d\t%s\t%s\t%s\t%0.3f\t%0.3f\t%0.3f\tkeymiss\n',...
            t,test_image{t},old_new_list(t),resp{t,1},resp{t,2},resp{t,3},rt(t,1),rt(t,2),rt(t,3)); 
            end;
        else,
            fprintf(fid,'%d\t%s\t%d\t%s\t%s\t%s\t%0.3f\t%0.3f\t%0.3f\tcell\n',...
           t,test_image{t},old_new_list(t),resp{t,1},resp{t,2},resp{t,3},rt(t,1),rt(t,2),rt(t,3)); 
        end;  
 
    end;  % end loop through trials	
      
end; %end cycle loop

Screen('TextSize',w,44);
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
    outfile=sprintf('%s_SM_%d%s_%s_%02.0f-%02.0f.mat',subject_code,list,FB_set,dt,d(4),d(5));

%  % create a data structure with info about the run
   run_info.initials=subject_code;
   run_info.date=date;
   run_info.outfile=outfile;
   run_info.problist=infilename;

   try,
  	save(outfile,'resp','rt','run_info', 'old_new_list', 'test_image', 'nts');
   catch,
	fprintf('couldn''t save %s\n saving to test.mat\n',outfile);
	save delayFB;
   end;		