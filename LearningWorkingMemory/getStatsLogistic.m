
%%%% Anne Collins
%%%% Brown University
%%%% December 2015
%%%% anne_collins@brown.edu

function getStats(si,sess)
% generates main results analysis for each subject - saves



st_b = nan(1,4);
perf = nan;
prop_no_answers_RLWM=[];
try
%analyze RLWMPST-learning
[st_b,perf,prop_no_answers_RLWM]=  analyzeRLWM(si,sess);
fl = 0;
catch
    disp('error in RLWM analysis');
fl = 1;
end
RLWM.performance = perf;
RLWM.prop_no_answers = prop_no_answers_RLWM;
RLWM.logistic = st_b;
RLWM.logistic_keys = {'fixed','set-size','delay','iterations'};


timestamp = clock;
save(['SummaryStats/S',num2str(si),'_Sess',num2str(sess)],'RLWM','RLWMPST','Effort','timestamp')
%save(['SummaryStats/S',num2str(si),'_Sess',num2str(sess)],'RLWM','RLWMPST','Effort','timestamp')

%% implement the file saving QC data.

printQC(RLWM,fl, si,sess);


dossier = 'GroupedExpeData\';
%dossier = 'GroupedExpeData/';
if exist([dossier,'RLWMPST_ID',num2str(si),'_Session',num2str(sess),'.mat'])
    load([dossier,'RLWMPST_ID',num2str(si),'_Session',num2str(sess),'.mat']);
end
if exist([dossier,'RLWMTraining_Su',num2str(si),'_Session',num2str(sess),'.mat'])
    load([dossier,'RLWMTraining_Su',num2str(si),'_Session',num2str(sess),'.mat']);
end
save(['SummaryStats\AllDataS',num2str(si),'_Sess',num2str(sess)])
%save(['SummaryStats/AllDataS',num2str(si),'_Sess',num2str(sess)])


end

%% Get the QC
function printQC(RLWM, fl,si,sess)

threshold_noA = .2;
threshold_EffortBias = .35;
threshold_PSTBias = .25;


if exist(['SummaryStats\QC_ID',num2str(si),'_Sess',num2str(sess),'.txt']);
%if exist(['SummaryStats/QC_ID',num2str(si),'_Sess',num2str(sess),'.txt']);
    fileattrib(['SummaryStats\QC_ID',num2str(si),'_Sess',num2str(sess),'.txt'],'+w');
    %fileattrib(['SummaryStats/QC_ID',num2str(si),'_Sess',num2str(sess),'.txt'],'+w');
end
fileID = fopen(['SummaryStats\QC_ID',num2str(si),'_Sess',num2str(sess),'.txt'],'w');
%fileID = fopen(['SummaryStats/QC_ID',num2str(si),'_Sess',num2str(sess),'.txt'],'w');
fprintf(fileID,['Subject ID ',num2str(si),' session ',num2str(sess),'\n\n']);



fprintf(fileID,['QC RLT task - learning: \n']);

if fl; test = 'FAIL';else test = 'PASS';end
fprintf(fileID,[test, ': Recovery of Learning file.\n']);

testnoA = RLWM.prop_no_answers<threshold_noA;
if testnoA; test = 'PASS';else test = 'FAIL';end
fprintf(fileID,[test, ': proportion of missed trials <',num2str(threshold_noA),'\n']);

testperf =RLWM.performance>.5;
if testperf; test = 'PASS';else test = 'FAIL';end
fprintf(fileID,[test, ': Performance > ',num2str(.5),'\n\n']);


fclose(fileID);
fileattrib(['SummaryStats/QC_ID',num2str(si),'_Sess',num2str(sess),'.txt'],'-w');
end

%% analyze RLWM learning phase

function [st_b,perf,prop_no_answers] = analyzeRLWM(si,sess)

dossier = 'GroupedExpeData\';
%dossier = 'GroupedExpeData/';

interaction = 0;
if interaction ==1
type =  {'Fixed','SetSize', 'delay','CorReps', '1x2', '1x3',  '2x3'};
else
type =  {'Fixed','SetSize', 'delay','CorReps'};
end


    
    [X,perf,prop_no_answers] = getData2(si,sess);
    %X(:,2) = min(1,2./X(:,2));
    % set size
    X(:,2) = -min(1,1./X(:,2));
    % delay
    X(:,3) = -1./X(:,3);
    % iterations
    X(:,4) = -1./X(:,4);
    
    
    z=repmat(nanmean(X(:,[2:4 9])),size(X,1),1);
    X(:,[2:4 9])=X(:,[2:4 9])-z;
    z=repmat(nanstd(X(:,[2:4 9])),size(X,1),1);
    X(:,[2:4 9])=X(:,[2:4 9])./z;
    
    % logistic regression (nS, time last correct, number of previous
    % corrects)
    %[B dev stats]=mnrfit([X(:,[2:4 9])],X(:,5)+1);%%%%logistic regression of perf depending on nS, delay, time
    if interaction 
    [B dev stats]=mnrfit([X(:,2:4) X(:,2).*X(:,3) X(:,2).*X(:,4) X(:,3).*X(:,4) X(:,2).*X(:,3).*X(:,4)],2-X(:,5));%%%%logistic regression of perf depending on nS, delay, time
    else
    [B dev stats]=mnrfit([X(:,[2:4])],2-X(:,5));%%%%logistic regression of perf depending on nS, delay, time
    end
    st_b=stats.beta';
    
end


function [MS,perf,prop_no_answers] = getData2(si,sess)

dossier = 'GroupedExpeData\';
%dossier = 'GroupedExpeData/';
    load([dossier,'RLWMPST_ID',num2str(si),'_Session',num2str(sess),'.mat'])
donnees = dataT;

blocks = donnees{end}.blocks;
T = length(blocks);

prop_no_answers = 0;
MS = [];
for epi=1:T
    if isempty(donnees{epi})
    else
    nS = blocks(epi);% number stimuli to learn
    D=donnees{epi};% episode's data
    X = D.Cor;
    W = D.winnings;
    prop_no_answers = prop_no_answers+length(find(X==-1));
    W(X==-1)=0;
    X(X==-1)=0;
            %X = dataT{bl}.Rew(T);
    Y = D.seq;
    Z = D.RT;
    
    Mat =-ones(1,nS);%%%1: time last correct ; 2: nb of previous corrects
    Mat = [Mat;zeros(4,nS)];
    for t = 1:length(X)
        st = Y(t);
        co = X(t);
        r = Z(t);
        
        MS = [MS;[si nS t-Mat(1,st) Mat(2,st) co Mat(3,st) r epi Mat(4,st)]];
        % sujet,  set size, delay, nb previous cor, cor, nb previous inc,
        % rt, block, prev rew
        if co ==1
            Mat(1,st) = t;
            Mat(5,st) = W(t)-Mat(4,st)/Mat(2,st);
            Mat(2,st) = Mat(2,st)+1;
            Mat(4,st) = W(t);%Mat(4,st) +W(t); %
        else
           %Mat(1,st) = t;
            Mat(3,st) = Mat(3,st)+1;
        end
        if co<1 &W(t)>0
            Mat
            co
            W(t)
            boum
        end
        Mat(4,st) = W(t); 
            
            
    end
    end
end
prop_no_answers = prop_no_answers/size(MS,1);
perf = nanmean(MS(:,4)>0);
MS(MS(:,4)==0,:)=[];


end


