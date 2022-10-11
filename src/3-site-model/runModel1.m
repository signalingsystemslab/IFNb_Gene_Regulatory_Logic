
% This file is to systematically explore the 8 State model. (5p) 
clear;

%% load exp matrix 
load('../data/exp_matrix_norm.mat') 

%% model linear eg : 1.enhanceosome 2. oR 3. IRF. 4. NF
tic
N= linspace(0,10,1000);
I = N;

temp=[ 1  0 0 0 0 0 0;
    1  0 1 1 1 1 1 ;
    1  1 1 0 1 1 1;
    1  0 0 1 0 1 1 ];
tlts = {'enhanceosome','OR','IRF','NF'};
figure;
for i =1:4
    subplot(2,2,i)
    pars = temp(i,:);
    m1 = model1(pars);
    m1 = calState(m1,N,I);
    m1 = calF(m1);
    plotCnorm(m1,N,I);
    title(tlts{i})
end
toc

%% check the objfunc 
pars = [ones(1)*.1 ones(1,6)]; 
[rmsd,rsqred,resid] = objfunc1(pars,exp_matrix,10,1)

%% optimization 
% obj=@(p) [~,~,r]= objfunc0([10^p tvec(j,:)],exp_matrix,10,1);
seed = 6;
rng(seed) % set seed for rands
ncpars = 1+6; 

numbPoints = 10^4; % number of vectors

parsSpace = rand(ncpars*numbPoints,1); 
parsSpace= reshape(parsSpace,numbPoints,ncpars);
parsSpace(:,1) = (parsSpace(:,1)*2); % 0 to 2
% parsSpace(:,1) = ((parsSpace(:,2)-0.5)*4);% -2 to 2
options = optimset('TolFun',1e-10,'TolX',1e-10,'Display','none'); 

lb=[ 0 0 0 0 0 0 0];ub=[ 2 1 1 1 1 1 1] ; 
rmsd = zeros(numbPoints,1); resid= zeros(numbPoints,8);parsFinal = zeros(numbPoints,ncpars);
tic
parfor i = 1:numbPoints
    % for i= 1:5
    [parsFinal(i,:), rmsd(i),resid(i,:)]= lsqnonlin(@obj1,parsSpace(i,:),lb,ub,options);
    if mod(i,numbPoints/10)==0
        disp(i)
    end
end
toc
%
save('../data/bestFit_model1_b.mat','parsFinal','rmsd','resid','seed')

