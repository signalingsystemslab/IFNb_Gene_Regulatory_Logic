% This file is to systematically explore the 8 State model. (5p) 
clear;

%% load exp matrix & plot exp matrix 
load('../data/exp_matrix_norm.mat') 


% start parallel pool
ncpu=20;
pc=parcluster('local');
pc.NumThreads=2;
parpool(pc,ncpu);

% %% check the objfunc 
% pars = [ones(1,2)*.1 ones(1,6)]; 
% [rmsd,rsqred,resid] = objfuncB4(pars,exp_matrix,10,1)


%% optimization 
% obj=@(p) [~,~,r]= objfunc0([10^p tvec(j,:)],exp_matrix,10,1);
seed =3;
rng(seed) % set seed for rands
ncpars = 2+6;
npts = 99; % number of synthetic points
nexp = length(exp_matrix.irf); % number of points in an experiment

numbPoints = 10^4; % number of vectors

parsSpace = rand(ncpars*numbPoints,1); 
parsSpace= reshape(parsSpace,numbPoints,ncpars);
parsSpace(:,1) = (parsSpace(:,1)*2); % 0 to 2 for KI2
parsSpace(:,2) = ((parsSpace(:,2)-0.5)*4);% -2 to 2 for C

% options = optimset('PlotFcns',{@optimplotx,@optimplotfunccount,@optimplotresnorm
% }, 'TolFun',1e-10,'TolX',1e-10); 

options = optimset('TolFun',1e-10,'TolX',1e-10,'Display','none',"Algorithm","levenberg-marquardt"); 

lb=[0 0 0 0 0 0 0 0];ub=[2 2 1 1 1 1 1 1] ; %first two pars are in log10 scale. 
rss = zeros(numbPoints,1); resid= zeros(numbPoints,nexp);parsFinal = zeros(numbPoints,ncpars);
%% Optimize experimental data
tic
fprintf("Starting optimization\n")
parfor i = 1:numbPoints
% for i = 1:5
%   Find local minima (of residuals) near each random parameter set
%     Optimize a vector by finding minimum of sum of squares (rmsd)
    [parsFinal(i,:), rss(i),resid(i,:),~,~]= lsqnonlin(@(p) obj(p, exp_matrix),parsSpace(i,:),lb,ub,options);
    if mod(i,numbPoints/10)==0
        disp(i)
    end
end
toc

% Calculate AIC and RMSD
aic = nexp*log(rss/nexp)+2*ncpars;
rmsd = sqrt(rss/nexp);

save('../data/bestFit_3site_b4.mat','parsFinal','rss','resid','rmsd','seed', ...
    "aic")
%% Optimize synthetic data
% About 25 mins
%  Load synthetic data
load('../data/synthetic_data.mat')
% 

fprintf("Starting optimization of synthetic data\n")
parsSyn = zeros(numbPoints,npts,ncpars);
resid_syn = zeros(numbPoints,npts,nexp);
rmsd_syn = zeros(numbPoints,npts);
rss_syn = rmsd_syn;
aic_syn = rmsd_syn;
tic
for k = 1:npts
    data_matrix.irf = new_points(1,:,k);
    data_matrix.nfkb = new_points(2,:,k);
    data_matrix.ifnb = new_points(3,:,k);
    
    parfor i = 1:numbPoints
%     for i = 1:5
    %     Optimize a vector by finding minimum of sum of squares (rmsd)
        [parsSyn(i,k,:), rss_syn(i,k),resid_syn(i,k,:),~,~]= lsqnonlin(@(p) obj(p, data_matrix),parsSpace(i,:),lb,ub,options);
        if mod(i,numbPoints/10)==0
            disp(i)
        end
    end
        rss = rss_syn(:,k);
        rmsd = sqrt(rss/nexp);
        aic = nexp*log(rss/nexp)+2*ncpars;

        rmsd_syn(:,k) = rmsd;
        aic_syn(:,k) = aic;

    if mod(k,10)==0
        fprintf("Finished %d points\n",k)
    end
end
toc

save('../data/bestFit_3site_b4_syndata.mat','parsSyn','rss_syn','rmsd_syn',"aic_syn","resid_syn")
%% Investigate synthetic data
fprintf("Saving syn data minimums\n")

load('../data/bestFit_3site_b4_syndata.mat')
min_rmsd_syn = zeros(npts,1);
min_aic_syn = min_rmsd_syn;
params_best_syn = zeros(npts,ncpars);
res_best_syn = zeros(npts,nexp);

for k = 1:npts
    rmsd = rmsd_syn(:,k);
    [rmsd, ind] = min(rmsd);
    min_rmsd_syn(k) = rmsd;
    min_aic_syn(k) = aic_syn(ind,k);
    params_best_syn(k,:) = parsSyn(ind,k,:);
    res_best_syn(k,:) = resid_syn(ind,k,:);
end

% Save only best values for each synthetic data set
save('../data/pars_3site_b4_syndata_small.mat','min_rmsd_syn','params_best_syn',"res_best_syn")
%% export data to csv files
fprintf("Exporting data to csv\n")

load('../data/bestFit_3site_b4.mat')
%  Get minimum exp data
[m,i] = min(rmsd);
par = parsFinal(i,:);
res = resid(i,:);
rmsd = rmsd(i);
aic = aic(i);
% combine with syn data
params = [par;params_best_syn];
rmsd = [rmsd;min_rmsd_syn];
aic = [aic;min_aic_syn];
res = [res;res_best_syn];
% label by dataset (experimental=0)
exp = [0, 1:npts]';
params=[params,exp];
rmsd=[rmsd,exp];
aic=[aic,exp];
res=[res,exp];

writematrix(params,'../data/ModelB4_parameters.csv')
writematrix(rmsd,'../data/ModelB4_rmsd.csv')
writematrix(aic,'../data/ModelB4_aic.csv')
writematrix(res,'../data/ModelB4_res.csv')
%% Functions
% In matlab local functions go at the end for some reason
function r=obj(p, exp_matrix)
    [~,~,r]=objfuncB4(p,exp_matrix,1,1);
end
