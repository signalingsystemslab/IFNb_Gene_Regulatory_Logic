% This file is to systematically explore the 8 State model. (5p) 
clear;

load ../data/exp_matrix_norm.mat

%% Initialize experimental data 
fprintf("Starting\n")
seed = 5;
rng(seed) % set seed for rands
ncpars = 6; 
ndata=length(exp_matrix.ifnb);

numbPoints = 10^4; % number of vectors

% Randomize 10k x6 parameters
% t0=0, t1-t6 are optimized, t7=1
% states: {'0','I1','I2','N','I1I2','I1N','I2N','I1I2N'}
parsSpace = rand(ncpars*numbPoints,1); 
parsSpace= reshape(parsSpace,numbPoints,ncpars);
% parsSpace(:,1) = (parsSpace(:,1)*4); % 0 to 4
% parsSpace(:,1) = ((parsSpace(:,1)-0.5)*4);% -2 to 2

% options = optimset('PlotFcns',{@optimplotx,@optimplotfunccount,@optimplotresnorm
% }, 'TolFun',1e-10,'TolX',1e-10); 

options = optimset('TolFun',1e-10,'TolX',1e-10,'Display','none'); 

lb=[ 0 0 0 0 0 0];ub=[ 1 1 1 1 1 1] ; 
rss = zeros(numbPoints,1); resid= zeros(numbPoints,ndata);parsFinal = zeros(numbPoints,ncpars);

%% Optimize experimental data
tic
fprintf("Starting optimization\n")
parfor i = 1:numbPoints
% for i = 1:5
%   Find local minima (of residuals) near each random parameter set

%     Optimize a vector by finding minimum of sum of squares (rmsd)
    [parsFinal(i,:), rss(i),resid(i,:),~,~]= lsqnonlin(@(p) obj(p, exp_matrix),parsSpace(i,:),lb,ub,options);
%     [parsFinal(i,:), rmsd(i),~,~]= fmincon(@(p)obj2(p, exp_matrix),parsSpace(i,:),[],[],[],[],lb,ub,[],options);
    if mod(i,numbPoints/10)==0
        disp(i)
    end
end
toc

% Calculate AIC
aic = ndata*log(rss/ndata)+2*ncpars;

save('../data/bestFit_3site_b1.mat','parsFinal','rss','resid','seed', ...
    "aic")
%% Optimize synthetic data

%% Functions
% In matlab local functions go at the end for some reason
function r=obj(p, exp_matrix)
%     p(1) = 10^p(1); %why?
    [~,~,r]=objfunc0(p,exp_matrix,1,1);
end


