%%  RLWM + PST - new version 2014 (Anne Collins))

%%%% Anne Collins
%%%% Brown University
%%%% November 2014
%%%% anne_collins@brown.edu

%%%% get inputs, run RLWM on those inputs.


function runRLWM(subject_id)


session = 1;
local_sujet = 1+rem(subject_id-1,10);


% get subjects' inputs
%cd('C:\Users\lncc\Dropbox\ExperimentsLNCCAnne\RLWMPST\InputsLSSt')
folder = '';%pwd;
folder = [folder,'InputsRLWM/']; % loads the optimized inputs
load([folder,'Randomization_S',num2str(local_sujet)]);
%cd ..
starttime=datetime;

blocks = Matrice.blocks;

% block/stimset identity assignation
stSets = Matrice.stSets;
% block/stim sequence assignation
stSeqs=Matrice.stSeqs;


% [1 2 3]-->keys mapping
Actions=Matrice.Actions;%%%%(modify: some keyboards have different keynumbers)
Actions(Actions == 13) = KbName('J');
Actions(Actions == 14) = KbName('K');
Actions(Actions == 15) = KbName('L');
% specific stimuli identity
stimuli=Matrice.stimuli;

% stimulus/correct action mapping
rules=Matrice.rules;

% run experiment with these inputs: RLWM phase 
dataT = FullRLWM(blocks,stSets,stSeqs,Actions,stimuli,rules,subject_id,local_sujet);

endlearningtime=datetime;

% save in subject specific file
save(['GroupedExpeData/RLWMPST_ID',num2str(subject_id)],'dataT','Matrice')



end
