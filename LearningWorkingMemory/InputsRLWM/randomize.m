clear all

for si=1:10
Matrice =[];
% block set sizes
blocks = [2 3 4 5 3 5 2 3 4 5];
p = randperm(3);
blocks = [blocks(1) blocks(p+1) blocks(5:7) blocks(p+7)];
Matrice.blocks = blocks;
%%

% how many stimuli per action; total
R= [1 1 0 2;
    0 1 1 2;
    2 0 1 3;% make sure that sometimes one action incorrect for all stim
    1 1 1 3;
    1 1 1 3;
    1 2 1 4;
    1 1 2 4;
    1 2 2 5;
    2 1 2 5;
    2 2 1 5];
p = randperm(3);
R = R(:,[p 4]);

realR = 0*R;
for ns=2:5
    T = find(blocks==ns);
    realR(T,:)=R(find(R(:,end)==ns),:);
end
R = realR;
Matrice.R = R;

%% rules

rules = [];
for b=1:length(blocks)
    thisrule=[];
    for a=1:3
        thisrule=[thisrule a*ones(1,R(b,a))];
    end
    rules{b}=thisrule;
end

Matrice.rules = rules;
%% stSets
Matrice.stSets = randperm(10);

%% Actions

Actions = 13:15;
Actions = Actions(randperm(3));
Matrice.Actions=Actions;
%% stimuli
stimuli = [];
for b=1:length(blocks)
    A=randperm(5);
    stimuli{b}=A(1:blocks(b));
end
Matrice.stimuli=stimuli;
%% stSeqs
reps=12;
for ns = 2:5
    seqprototype{ns}=createstimsequence(reps,ns);
end

stSeqs=[];
for b=1:length(blocks)
    thisseq=seqprototype{blocks(b)};
    A=randperm(blocks(b));
    thisseq=A(thisseq);
    stSeqs{b}=thisseq;
    
end
Matrice.stSeqs=stSeqs;


save(['Randomization_S',num2str(si)],'Matrice')
end