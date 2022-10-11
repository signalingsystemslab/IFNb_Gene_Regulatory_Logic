% This file is to systematically explore the 8 State model. (5p) 
clear;

%% load exp matrix

load('../data/exp_matrix_norm.mat') 

%% model linear eg : 1.enhanceosome 2. oR 3. IRF. 4. NF
tic
N= linspace(0,10,1000);
I = N;

temp=[ 1 0 0;
    1 1 1 ;
    1 1 0;
    1 0 1 ];
tlts = {'enhanceosome','OR','IRF','NF'};
figure;
for i =1:4
    subplot(2,2,i)
    pars = temp(i,:);
    m1 = model0(pars);
    m1 = calState(m1,N,I);
    m1 = calF(m1);
    plotCnorm(m1,N,I);
    title(tlts{i})
end
toc


%% check the objfunc 
pars = [ones(1,1)*.1 ones(1,2)]; 
% pars =[parsSpace(b,:) 0 0 0 1 0 1];
[rmsd,rsqred,resid] = objfunc0(pars,exp_matrix,10,1)
plot(resid,'o-')
%% let's explore the parameter space. 
ncpars = 1; 
rng(3) % set seed for rand
numbPoints = 10^6; % number of vectors
tvec = npermutek([0 1],2);
ntvec = size(tvec,1);

%
parsSpace = rand(ncpars*numbPoints,1); 
parsSpace= reshape(parsSpace,numbPoints,ncpars);

parsSpace = 10.^((parsSpace-0.5)*8);

%%
% rmsd =cell(ntvec,1); 
% resid=cell(ntvec,1);

rmsd = zeros(numbPoints,1); resid= zeros(numbPoints,7);
%
tic
for j = 2:4%1:ntvec
   parfor i = 1:numbPoints
        [rmsd(i),~,resid(i,:)]= objfunc0([parsSpace(i,:) tvec(j,:)],...
            exp_matrix,10,1);
        if mod(i,numbPoints/10)==0
             disp(i)
         end
   end
    disp(j)
    save(['../data/model0_lin10_normb',num2str(j),'.mat'],...
        'parsSpace','resid','rmsd')
end

toc 



%% find all minmals and save data
nfiles = 4; 
minRmsd = zeros(1,nfiles);
minRmsdind = zeros(1,nfiles);
rmsdQuantile = cell(1,4);
for jth = 1:nfiles
    load(['../data/model0_lin10_normb',num2str(jth),'.mat']);
    [minRmsd(jth) minRmsdind(jth)] =   min(rmsd);
    rmsdQuantile{jth} = rmsd(rmsd<prctile(rmsd,10));
end
%
save('../data/model0_rmsd_all.mat','minRmsd','minRmsdind','rmsdQuantile')




