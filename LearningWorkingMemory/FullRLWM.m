%%  RLWM + PST - new version 2014 (Anne Collins))

%%%% Anne Collins
%%%% Brown University
%%%% November 2014
%%%% anne_collins@brown.edu

function dataT=FullRLWM(blocks,stSets,stSeqs,Actions,stimuli,rules,subject_id,local_sujet)

directory = 'GroupedExpeData';

debug = 0;
load debugstate

timing = [2 .2 .75 .5];% stim presentation, choice-FB, FB, ITI
%debug = 0;
nA = 3;
%load datatmp
dataT=[];

% create input file to be saved within subject's output data
Entrees=[];
Entrees.local_sujet = local_sujet;
Entrees.blocks=blocks;
Entrees.stSets=stSets;
Entrees.stSeqs=stSeqs;
Entrees.Actions=Actions;
Entrees.stimuli=stimuli;
Entrees.rules=rules;
dataT{length(blocks)+1}=Entrees;

text{1} = 'Block ';
text{2} = 'Take some time to look at the images for this block.';
text{3} = '[Press space to continue.]';
text{4} = '';
text{5} = '0';
text{6} = 'End of block ';
text{7} = 'End of this experiment!';
text{8} = 'No valid answer';

% --------------- 
% open the screen
% ---------------


time = 0;

try
    
        
    
    % get screen parameters
    Screen('Preference', 'SkipSyncTests',1);
    if debug 
    screenRect = [0,0,1250,800]; % screen for debugging
    else
    screenRect = []; % full screen
    end
    [w, rect] = Screen('OpenWindow', 0, 0,screenRect,32,2);
    HideCursor;	% Hide the mouse cursor
    %ListenChar(2);
    
    RLWMtraining(subject_id,w,rect,Actions);
    
    %[w, rect] = Screen('OpenWindow', 0, 0,screenRect,32,2);
    %HideCursor;	% Hide the mouse cursor
    
    % create stim sample display boxes
    x=rect(3)/7;
    x=min(x,rect(4)/5);
    boitesPresentation=[1*x 1*x 2*x 2*x;
                        3*x 1*x 4*x 2*x;
                        5*x 1*x 6*x 2*x;
                        1*x 3*x 2*x 4*x;
                        3*x 3*x 4*x 4*x;
                        5*x 3*x 6*x 4*x];
    
    %HideCursor;	% Hide the mouse cursor
    %ListenChar(2);
    
%    Screen('TextFont', w , 'Arial');
%    Screen('TextSize', w, 32 );
    

    Tinitial=GetSecs;
    if debug
        torun = [1 length(blocks)];
    else
        torun = [1:length(blocks)];
    end
    
       
    %Run experiment for 13 blocks
    for b=torun
        data=[];
        
        % Get stim set size
        nS=blocks(b);%%%Combien d'images
        % Create a matrix to store the stimuli
        SMat=zeros(300,300,3,nS);
        % get the specific stimuli numbers
        sordre=stimuli{b};%%%Quelles images dans la famille
        % get the specific stimulus type
        Type=stSets(b);%%%%Famille d'images
        % load stimuli and store them in matrix SMat for display
        for i=1:nS
            load(['imagesRLWM\images',num2str(Type),'\image',num2str(sordre(i))]);%%%load des stimuli
            SMat(:,:,:,i)=X;
        end
        
        %%%Get stim sequence for block b
        stimseq=stSeqs{b};
        taille=length(stimseq);
        
        %%%Get correct action sequence for block b, using predifined rules
        actionseq=0*stimseq;
        TS=rules{b};
        for i=1:nS
            actionseq(stimseq==i)=TS(i);
        end
        TMin=nS*3*3;
        
        
        % get screen parameters
%         Screen('Preference', 'SkipSyncTests',1);
%         if debug 
%         screenRect = [0,0,1250,800]; % screen for debugging
%         else
%         screenRect = []; % full screen
%         end
%         [w, rect] = Screen('OpenWindow', 0, 0,screenRect,32,2);
%         HideCursor;	% Hide the mouse cursor
        %ListenChar(2);
    
        % Stim sets presentation in previously built boxes
        Screen('FillRect', w,0);
        Screen('TextColor',w,[255 255 255]);
        textI = [text{1},num2str(b),'\n\n\n',text{2},'\n\n\n',text{3}];        
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
               
        WaitSecs(.2);
        
        % wait for key press to begin block
        %WaitSecs(2)
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
        
        while t<tmax
            t=t+1;
            time = time+1;
             sti=stimseq(t);
            acor=actionseq(t);
            imaget=SMat(:,:,:,sti);
            
            [RT,code,cor]=singleTrial(w,rect,Actions,imaget,acor,timing);
            

            timeseq(t) = time;
            DCor(t)=cor;
            DRT(t)=RT;
            DCode(t)=code;
        end % ending sessions

        % build output structure with all info in
        data.temps=t;
        data.Cor=DCor;
        data.RT=DRT;
        data.Code=DCode;
        data.seq=stimseq;
        data.actionseq=actionseq;
        data.timeseq = timeseq;
        dataT{b}=data;
        
        % Give block feedback on time saved
        Screen('TextSize', w, 32 );
        Screen('FillRect', w,0)
        textI = [text{6}, num2str(b),...
            '\n\nYou chose correctly ',num2str(sum(DCor==1)),...
            ' times out of ',num2str(t),' trials!',...
            '\n\n\n\n\n\n\n',text{3}];
        DrawFormattedText(w,textI,'center','center');
        Screen('Flip', w);kdown=0;
        
        
        while kdown==0;
            kdown=KbCheck;
        end
        Screen('FillRect', w, 0)
        Screen('Flip', w);
        WaitSecs(2);
        save([directory,'/RLWMPST_ID',num2str(subject_id)],'dataT')
        
    end
    TFinal=GetSecs;
    
    % Give global feedback
    Duree=TFinal-Tinitial;
    minutes=floor(Duree/60);
    secondes=floor(Duree)-60*minutes;
    Screen('FillRect', w,0)
    textI = text{7};
    DrawFormattedText(w,textI,'center','center');
    Screen('Flip', w);
    WaitSecs(2); 
        
        
    ShowCursor;
    %ListenChar(0);
    Screen('CloseAll');
    Snd('Close')
catch
    ShowCursor;
    %ListenChar(0);
    Screen('CloseAll');
    Snd('Close')
    rethrow(lasterror);
end




end

