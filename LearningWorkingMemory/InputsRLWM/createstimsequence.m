function seq = createstimsequence(reps,ns)

%% create sequence with controlled delays
% reps = 8;
% ns=6;

seq = repmat(1:ns,1,reps);

beta=5;

figure(1);
criterion=false;
ps=[];
while ~criterion
    delays = 1+floor(ones(1,2*ns-1)*reps/2);
    count = reps*ones(1,ns)+1;
    seq=1:ns;
    last = 1:ns;
    for t=ns+1:ns*(reps+1)
        Q=[];
        for i=1:ns
            Q(i)=delays(t-last(i))+count(i);
            L(i)=t-last(i);
        end
        if max(L)==size(delays,2)
            [~,choice]=max(L);
        else
        softmax=exp(beta*Q);softmax = softmax./sum(softmax);
        ps=[0 cumsum(softmax)];
        choice = find(ps<rand);choice=choice(end);
        end
        seq(t)=choice;
        last(choice)=t;
        delays(L(choice))=delays(L(choice))-1;
        count(choice)=count(choice)-1;
    end
    
    
delay=[];
last=zeros(1,ns);
alldelays=[];
dseq=[];
for t=1:length(seq)
    if last(seq(t))>0
        alldelays = [alldelays t-last(seq(t))];
        dseq=[dseq seq(t)];
    end
    last(seq(t))=t;
end
ds=unique(alldelays);
distr=[];
for d=ds
    distr(d)= sum(alldelays==d);
end
%distr=distr/sum(distr);
[h,p,st] = chi2gof(alldelays,'edges',[1:length(distr)+1]-.5,...
    'expected',mean(distr)*ones(1,length(distr)));

figure(1)
plot(distr,'o-')
criterion = p>.05;

criterion = (max(distr)-min(distr))<2;
for d=ds
    for i=1:ns
        distr(i,d)=sum(alldelays==d & dseq==i);
    end
end
hold on
distr = sum(distr(:,:));
plot(distr','o-')
hold off

criterion =criterion & ((max(distr)-min(distr))<2);
end

end
%% create sequence with controlled delays