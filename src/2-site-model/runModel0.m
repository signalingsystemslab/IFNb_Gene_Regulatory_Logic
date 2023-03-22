% This file is to systematically explore the 2 site model. (5p) 
clear;

%% Set up
% load data
load('../data/exp_matrix_norm.mat') 
fprintf("try 3\n")
% start parallel pool
ncpu=20;
pc=parcluster('local');
pc.NumThreads=2;
parpool(pc,ncpu);
% %% Generate synthetic data
npts = 99; % number of synthetic points
nexp = length(exp_matrix.irf); % number of points in an experiment
new_points = zeros(3,nexp,npts);
data_points = exp_matrix;
rng(2)
%    stdev=sterr is 10% of maximum signal
irf_dev = max(exp_matrix.irf)/10;
irf_var = irf_dev^2;
nfkb_dev = max(exp_matrix.nfkb)/10;
nfkb_var = nfkb_dev^2;
Sigma = [irf_var 0; 0 nfkb_var];

parfor k = 1:npts
    mu = [exp_matrix.irf' exp_matrix.nfkb'];
    X = zeros(3,nexp);
    % sample until both constraints are met
    while (X(1,1) <= X(1,3)) || (X(1,6) <= X(1,7))
%         fprintf("x1: %.5f, x3: %.5f \n x6: %.5f, x7: %.5f \n", ...
%             X(1,1), X(1,3), X(1,6), X(1,7))
        for j = 1:nexp
            s = mvnrnd(mu(j,:),Sigma,1);
            s = max(s,0);
            s = min(s,1);
            X(1:2,j) = s;
            X(3,j) = exp_matrix.ifnb(j);
        end
    end
    % add sample to new points matrix
    new_points(:,:,k) = X;
end

data_points.irf = [data_points.irf,new_points(1,:)]; % not needed
data_points.nfkb = [data_points.nfkb,new_points(2,:)];
data_points.ifnb = [data_points.ifnb,new_points(3,:)];

%  export data to csv
krep = [repelem(0,nexp)];
for k=1:npts
    krep = [krep, repelem(k, nexp)];
end
M = [data_points.irf', data_points.nfkb', data_points.ifnb', krep'];
M = ["IRF", "NFkB", "IFNb", "exp";M];
writematrix(M,'../data/syn_data.csv')


ndata = length(data_points.irf); %(npts+1)*7
save('../data/synthetic_data.mat','data_points','new_points','ndata')
% load('../data/synthetic_data.mat')
fprintf("Done generating synthetic data \n")

%% Build parameters set 
ncpars = 1; 
rng(3) % set seed for rand
numbPoints = 10^6; % number of vectors
tvec = npermutek([0 1],2);
ntvec = size(tvec,1);

% %
% parsSpace = rand(ncpars*numbPoints,1); 
% parsSpace= reshape(parsSpace,numbPoints,ncpars);
% 
% parsSpace = 10.^((parsSpace-0.5)*8); %1e-4 to 1e4
% parsSpace = sort(parsSpace);
% save('../data/pars_model2site.mat','parsSpace')
load('../data/pars_model2site.mat')

%% Minimize RMSD
rmsd =cell(ntvec,1); 
resid=cell(ntvec,1);

rmsd = zeros(numbPoints,1); resid= zeros(numbPoints,7);

tic
% save all RMSD values for real data
fprintf("Finding RMSD for exp data \n")
for j = 1:ntvec
   parfor i = 1:numbPoints
        [rmsd(i),~,resid(i,:)]= objfunc0([parsSpace(i,:) tvec(j,:)],...
            exp_matrix,1,1);
        if mod(i,numbPoints/10)==0
             disp(i)
         end
   end
    fprintf("Finished %d models\n", j)
    save(['../data/model0_lin10_normb',num2str(j),'.mat'],...
        'parsSpace','resid','rmsd')
end

% find minimum RMSD for syn data
fprintf("Finding RMSD for syn data \n")
syn_mins = zeros(ntvec,2,npts); %min rmsd, best C param

% model, rmsd-param, dataset, params
rmsd_syn = zeros(ntvec,2,npts,numbPoints);
for k = 1:npts
    data_matrix.irf = new_points(1,:,k);
    data_matrix.nfkb = new_points(2,:,k);
    data_matrix.ifnb = new_points(3,:,k);
    
    for j = 1:ntvec
        t = tvec(j,:);
        parfor i = 1:numbPoints
            p = parsSpace(i,:);
            [r,~,~]= objfunc0([p t],data_matrix,1,1);
            rmsd_syn(j,:,k,i) = [r, p];
            if mod(i,numbPoints/10)==0
                disp(i)
            end
        end
        [m,i] = min(rmsd_syn(j,1,k,:));
        syn_mins(j,1,k) = m;
        syn_mins(j,2,k) = parsSpace(i,:);
    end
    fprintf("Finished %d points\n",k)
end
save("../data/minRMSD_synthetic.mat",'syn_mins','rmsd_syn', '-v7.3')

% end parallel
poolobj = gcp('nocreate');
delete(poolobj);
toc 

%% find all minimals and save data
nfiles = 4; 
minRmsd = zeros(1,nfiles);
minRmsdind = zeros(1,nfiles);
minRmsdParam = zeros(ncpars,nfiles);
rmsdQuantile = cell(1,nfiles);
for jth = 1:nfiles
    load(['../data/model0_lin10_normb',num2str(jth),'.mat']);
    [minRmsd(jth), minRmsdind(jth)] =   min(rmsd);
    rmsdQuantile{jth} = rmsd(rmsd<prctile(rmsd,10));
    param = parsSpace(minRmsdind(jth),:);
    minRmsdParam(:,jth) = param;
end
%
tnames = ["AND", "NFkB", "IRF", "OR"];
save('../data/model0_rmsd_all.mat','minRmsd','minRmsdind','rmsdQuantile', 'minRmsdParam',"tnames")

% tvec
% I N
% 0 0 AND - j=1
% 0 1 NFkB - j=2
% 1 0 IRF - j=3
% 1 1 OR - j=4

%  Export to load in R
M = ["minRMSD", "bestC","model";minRmsd', minRmsdParam',tnames'];
writematrix(M,'../data/exp_data_mins.csv')

trep = [];
for j=1:ntvec
    trep = [trep, repelem(tnames(j), npts)];
end
sm2 = permute(syn_mins, [3 1 2]);
M = reshape(sm2,[npts*ntvec,2]);
M = horzcat(M, trep');
M = ["minRMSD", "bestC","model"; M];
writematrix(M,'../data/syn_data_mins.csv')

%% Explore minimals and maximals
% load("../data/minRMSD_synthetic.mat")
labs = ["IRF_bestRMSD","IRF_worstRMSD","IRF_highC","IRF_lowC","AND_highC","AND_lowC"];
cols = ["Dataset_number","RMSD","C_Value"];
vals = [];

% syn_mins_IRF = syn_mins(3,:,:);
% syn_mins_AND = syn_mins(1,:,:);
[r,i] = max(squeeze(syn_mins(3,1,:)));
vals = [vals; i, r, syn_mins(3,2,i)];
[r,i] = min(squeeze(syn_mins(3,1,:)));
vals = [vals; i, r, syn_mins(3,2,i)];

[p,i] = max(squeeze(syn_mins(3,2,:)));
vals = [vals; i, syn_mins(3,1,i), p];
[p,i] = min(squeeze(syn_mins(3,2,:)));
vals = [vals; i, syn_mins(3,1,i), p];

[p,i] = max(squeeze(syn_mins(1,2,:)));
vals = [vals; i, syn_mins(1,1,i), p];
[p,i] = min(squeeze(syn_mins(1,2,:)));
vals = [vals; i, syn_mins(1,1,i), p];

save("../data/interesting_values.mat","vals", "labs","cols")


fprintf("Done \n")