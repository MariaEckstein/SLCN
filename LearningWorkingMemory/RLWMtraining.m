%%  RLWM + PST - new version 2014 (Anne Collins))

%%%% Anne Collins
%%%% Brown University
%%%% November 2014
%%%% anne_collins@brown.edu

%%%% get inputs, run RLWM on those inputs.

%%%% Training Code


function RLWMtraining(subject_id,w,rect,Actions)
debug = 1;
load debugstate

timing = [7 .2 .75 .5];% stim presentation, choice-FB, FB, ITI
rand('twister',sum(100*clock));
nA=3;


% --------------- 
% open the screen
% ---------------
% change 'block' to 'level'
text{1} = 'Level ';
text{2} = 'Take some time to look at the images for this level.';
text{3} = ' '; % [Press any button to continue.]
text{4} = '';
text{5} = '0';
text{6} = 'End of level ';
text{8} = 'No valid answer';
text{9} = 'End of training.';

time=0;
try
    
    HideCursor;	% Hide the mouse cursor
    %ListenChar(2);
    
    
    % Create display boxes
    x=rect(3)/7;
    x=min(x,rect(4)/5);
    boitesPresentation=[
                        3*x 1*x 4*x 2*x;
                        3*x 3*x 4*x 4*x];
    
   
    Tinitial=GetSecs;
    
        % Get stim set size
    nS=2;
    % Create a matrix to store the stimuli
    SMat=zeros(300,300,3,nS);
    % load stimuli and store them in matrix SMat for display
    for i=1:nS
        load(['imagesRLWM/PracticeImages/image',num2str(i)]);%%%load des stimuli
        SMat(:,:,:,i)=X;
    end
        
    % Give instructions
    InstructionsLSSt_controller(w,rect,Actions,SMat);
    
    for i=1:nS
        load(['imagesRLWM/TrainingImages/image',num2str(i)]);%%%load des stimuli
        SMat(:,:,:,i)=X;
    end
    %%%Get stim sequence for block b
    stimseq=Shuffle(repmat([1:2],1,25));
    actionseq=stimseq;
    taille=length(stimseq);

    % Stim sets presentation in previously built boxes
    Screen('FillRect', w,0)
    Screen('TextColor',w,[255 255 255]);
    textI = ['Beginning of practice\n\n\n',text{2},'\n\n\n',text{3}];        
    DrawFormattedText(w,textI,'center','center');
    Screen('Flip', w);

    WaitSecs(.2);
    kdown=0;
    while kdown==0;
        kdown=KbCheck;
    end

    Screen('FillRect', w, 0)
    for tt=1:nS
        Screen(w,'PutImage',SMat(:,:,:,tt),boitesPresentation(tt,:))
    end

    Screen('Flip', w);

    % wait for key press to begin block
    WaitSecs(2)
    kdown=0;
    while kdown==0;
        kdown=KbCheck;
    end
    Screen('FillRect', w, 0)
    Screen('Flip', w);
    WaitSecs(2)
    Screen('TextSize', w, 75 );


    % set up criterion
    t=0;
    DCor=[];
    DCode=[];
    DRT=[];
    timeseq = [];

    if debug;tmax = 10; else tmax = taille;end

    critere=0;
    while t<tmax & critere<1
        t=t+1;
        time = time+1;
        %get trial's stimulus
        sti=stimseq(t);
        acor=actionseq(t);
        imaget=SMat(:,:,:,sti);

        [RT,code,cor]=singleTrial(w,rect,Actions,imaget,acor,timing);


        timeseq(t) = time;
        DCor(t)=cor;
        DRT(t)=RT;
        DCode(t)=code;

        if t>15
            perf = mean(DCor(t-9:t));
            critere = perf>=.8;
        else
            critere = 0;
        end
    end
    data = [];
    data.temps=t;
    data.Cor=DCor;
    data.RT=DRT;
    data.Code=DCode;
    data.seq=stimseq;
    data.actionseq=actionseq;
    data.timeseq = timeseq;
    data.buttonz = Actions;

    directory = 'GroupedExpeData';
    save([directory,'/RLWMTraining_Su',num2str(subject_id)],'data');

    % give block feedback 
    Screen('TextFont',w,'Arial');
    Screen('TextSize', w, 32 );
    Screen('FillRect', w,0)
    textI = ['End of practice! \n\n\n\n\n\n\n',...
        text{3}];
    DrawFormattedText(w,textI,'center','center');

    Screen('Flip', w);kdown=0;
    while kdown==0;
        kdown=KbCheck;
    end
    Screen('FillRect', w, 0);
    Screen('Flip', w);
    WaitSecs(2);

    TFinal=GetSecs;
    
    Duree=TFinal-Tinitial;
    minutes=floor(Duree/60);
    secondes=floor(Duree)-60*minutes;
    Screen('FillRect', w,0)
    Screen(w,'DrawText',text{9},(rect(3)/2)-150,rect(4)/2-100,255);
    %Screen(w,'DrawText',['It lasted ',num2str(minutes),' min, ',num2str(secondes),' sec.'],(rect(3)/2)-150,rect(4)/2,255);
        
%     ShowCursor;
    %ListenChar(0);
%     Snd('Close')
%     Screen('CloseAll');
catch
    ShowCursor
    %ListenChar(0);
    Screen('CloseAll');
    Snd('Close')
    rethrow(lasterror);
end




end

    