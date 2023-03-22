% This file is to systematically explore the 8 State model. (5p) 
clear;

%% load exp matrix & plot exp matrix 
load('../data/exp_matrix_norm.mat') 



%% check the objfunc 
pars = [ones(1,2)*.1 ones(1,6)]; 
[rmsd,rsqred,resid] = objfunc3(pars,exp_matrix,10,1)


%% optimization 
% obj=@(p) [~,~,r]= objfunc0([10^p tvec(j,:)],exp_matrix,10,1);
rng(3) % set seed for rands
ncpars = 2+6; 

numbPoints = 10^4; % number of vectors

parsSpace = rand(ncpars*numbPoints,1); 
parsSpace= reshape(parsSpace,numbPoints,ncpars);
parsSpace(:,1) = (parsSpace(:,1)*2); % 0 to 2
parsSpace(:,2) = (parsSpace(:,2)*2);% 0 to 2

% options = optimset('PlotFcns',{@optimplotx,@optimplotfunccount,@optimplotresnorm
% }, 'TolFun',1e-10,'TolX',1e-10); 

options = optimset('TolFun',1e-10,'TolX',1e-10,'Display','none'); 

lb=[0 0 0 0 0 0 0 0];ub=[2 2 1 1 1 1 1 1] ; %first two pars are in log10 scale. 
rmsd = zeros(numbPoints,1); resid= zeros(numbPoints,8);parsFinal = zeros(numbPoints,ncpars);
tic
parfor i = 1:numbPoints
    % for i= 1:5
    [parsFinal(i,:), rmsd(i),resid(i,:)]= lsqnonlin(@obj3,parsSpace(i,:),lb,ub,options);
    if mod(i,numbPoints/10)==0
        disp(i)
    end
end
toc
%

save('./data/bestFit_model3_b.mat','parsFinal','rmsd','resid')

