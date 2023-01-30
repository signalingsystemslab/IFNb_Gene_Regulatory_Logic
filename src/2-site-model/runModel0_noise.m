%% Repeat Model0 two-state model optimization but with some noise
% This file is to explore the 2 site model with noise around data.
clear;

%% load exp matrix

load('../data/exp_matrix_norm.mat')
data_points = exp_matrix;
rng(2)
%    stdev=sterr is 10% of maximum signal
irf_dev = max(exp_matrix.irf)/10;
irf_var = irf_dev^2;
nfkb_dev = max(exp_matrix.nfkb)/10;
nfkb_var = nfkb_dev^2;
    
for i = 1:length(exp_matrix.irf)
%     multivariate normal distribution
    mu = [exp_matrix.irf(i) exp_matrix.nfkb(i)];
    Sigma = [irf_var 0; 0 nfkb_var];
    X = mvnrnd(mu,Sigma,99);
    X = max(X,0);
    X = min(X,1);
    data_points.irf = [data_points.irf,X(:,1)'];
    data_points.nfkb = [data_points.nfkb,X(:,2)'];
    data_points.ifnb = [data_points.ifnb,repelem(exp_matrix.ifnb(i),99)];
end

ndata = length(data_points.irf);

% verify data points look good
figure;
scatter(data_points.irf*10,data_points.nfkb*10,20,data_points.ifnb,"filled")
ylabel('NFkb');xlabel('IRF');
%% check the objfunc 
pars = [ones(1,1)*.1 ones(1,2)]; 
% pars =[parsSpace(b,:) 0 0 0 1 0 1];
[rmsd,rsqred,resid] = objfunc0(pars,data_points,10,1);
plot(resid,'o-')
%% best 10,000 parameters from observations 
ncpars = 1; 
% rng(3) % set seed for rand
% numbPoints = 10^6; % number of vectors
% tvec = npermutek([0 1],2);
% ntvec = size(tvec,1);
% 
% %
% parsSpace = rand(ncpars*numbPoints,1); 
% parsSpace= reshape(parsSpace,numbPoints,ncpars);
% 
% parsSpace = 10.^((parsSpace-0.5)*8);
% parsSpace = sort(parsSpace);
npars = 10000;
pars = zeros(nfiles,npars);
nfiles = 4;


for jth = 1:nfiles
    load(['../data/model0_lin10_normb',num2str(jth),'.mat']);
    [~,indexes] = sort(rmsd);
     % add 10k best parameters
%     display(indexes(1:5))
    pars(jth,:) = parsSpace(indexes(1:npars));
%     pars=vertcat(pars, parsSpace(indexes(1:npars)));
%     display(parsSpace(1:5))
end

parsSpace = reshape(pars,npars*nfiles,ncpars);
parsSpace = sort(unique(parsSpace));
numbPoints = length(parsSpace);
%%
% rmsd =cell(ntvec,1); 
% resid=cell(ntvec,1);

rmsd = zeros(numbPoints,1); resid= zeros(numbPoints,ndata);
%
tic
for j = 1:ntvec
   parfor i = 1:numbPoints
        [rmsd(i),~,resid(i,:)]= objfunc0([parsSpace(i,:) tvec(j,:)],...
            data_points,10,1);
        if mod(i,numbPoints/10)==0
             disp(i)
         end
   end
    disp(j)
    save(['../data/model0_2statenoise_b',num2str(j),'.mat'],...
        'parsSpace','resid','rmsd')
end

toc 



%% find all minimals and save data
nfiles = 4; 
minRmsd = zeros(1,nfiles);
minRmsdind = zeros(1,nfiles);
minRmsdParam = zeros(ncpars,nfiles);
rmsdQuantile = cell(1,nfiles);
for jth = 1:nfiles
    load(['../data/model0_2statenoise_b',num2str(jth),'.mat']);
    [minRmsd(jth), minRmsdind(jth)] =   min(rmsd);
    rmsdQuantile{jth} = rmsd(rmsd<prctile(rmsd,10));
    param = parsSpace(minRmsdind(jth),:);
    minRmsdParam(:,jth) = param;
end
%
save('../data/model0_rmsd_2statenoise.mat','minRmsd','minRmsdind','rmsdQuantile', 'minRmsdParam')

tnames = ["t3", "t2", "t1", "t4"];
M = [tnames; minRmsd; minRmsdParam];
save('../data/model0_minimums_2statenoise.mat', 'M')

%% Plot minimum RMSD and corresponding parameter

subplot(2,1,1)
bar(minRmsd)
ylabel('RMSD');xlabel('AND,NFkB,IRF,OR');
subplot(2,1,2)
bar(minRmsdParam)
ylabel('C at min RMSD');xlabel('AND,NFkB,IRF,OR');

%% model best-fitting eg : 1.enhanceosome 2. oR 3. IRF. 4. NF
tic
N= linspace(0,10,1000);
I = N;
% dat = [exp_matrix.irf; exp_matrix.nfkb; exp_matrix.ifnb];

% min param for each model + model matrix
temp=horzcat(minRmsdParam', tvec);
tlts = {'AND','NFkB','IRF','OR'};
figure;
for i =1:4
    subplot(2,2,i)
    pars = temp(i,:);
    m1 = model0(pars);
    m1 = calState(m1,N,I);
    m1 = calF(m1);
    plotCnorm(m1,N,I, data_points);
    title(tlts{i})
end
toc


