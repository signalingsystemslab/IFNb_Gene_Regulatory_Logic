% runModel0 without noisefinds minimum RMSD of experimental data and
% simulation. This script bootstraps the data to find error for minimum
% RMSD.
clear;

%% Load data
load('../data/exp_matrix_norm.mat') 


%% Initialize params
% load 10000 best param values
ncpars = 1; 
npars = 10000;
pars = zeros(nfiles,npars);
nfiles = 4;

fprintf("Loading params \n")
for jth = 1:nfiles
    load(['../data/model0_lin10_normb',num2str(jth),'.mat']);
    [~,indexes] = sort(rmsd);
    pars(jth,:) = parsSpace(indexes(1:npars));
end

parsSpace = reshape(pars,npars*nfiles,ncpars);
parsSpace = sort(unique(parsSpace));
numbPoints = length(parsSpace);
fprintf("%d params \n",numbPoints)

%% Bootstrap RMSD to get error bars
ndata = length(exp_matrix.irf);
n_resample = 10000;
minVals.rmsd=zeros(n_resample,ntvec);
minVals.param=zeros(n_resample,ntvec);
% minVals = [];

fprintf("Starting \n")
tic

for r = 1:n_resample
    rng(r)
    s = randsample(7,7,true);
    exp_sample.irf = exp_matrix.irf(s);
    exp_sample.nfkb = exp_matrix.nfkb(s);
    exp_sample.ifnb = exp_matrix.ifnb(s);
 
    if mod(r,n_resample/10)==0
         fprintf("Resampling #%d \n",r);
    end
   
    rmsd = zeros(numbPoints,1);
    for j = 1:ntvec
        t_temp = tvec(j,:);
       parfor i = 1:numbPoints
            [rmsd(i),~,~]= objfunc0([parsSpace(i,:) t_temp],...
                exp_sample,10,1);
       end

       if mod(r,n_resample/10)==0
         fprintf("Finished model %d \n",j);
       end

       [minR, minI] = min(rmsd);
       
%        minVals = [minVals; minR, minI];
       minVals.rmsd(r,j) = minR;
       minVals.param(r,j) = parsSpace(minI);
    end
    
    

end
toc

save("../data/model0_bootstrap_mins.mat", "minVals")
fprintf("saved data. \n")