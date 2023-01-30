% This file is to systematically explore the 8 State model. (5p) 
clear;

%% load exp matrix

load('../data/exp_matrix_norm.mat') 

% verify data points look good
figure;
scatter(exp_matrix.irf,exp_matrix.nfkb,50,exp_matrix.ifnb,"filled")
cmap = readmatrix("../data/colormap.csv");
colormap(cmap);
colorbar;
ylabel('NFkb');xlabel('IRF');
ax = gca;
exportgraphics(ax,'exp_data_noscale.png');

%% model linear eg : 1.enhanceosome 2. oR 3. IRF. 4. NF
tic
N= linspace(0,1,1000);
I = N;


temp=[ 1 0 0;
    1 1 1 ;
    1 1 0;
    1 0 1 ];
tlts = {'AND','OR','IRF','NF'};
figure;
for i =1:4
    subplot(2,2,i);
    pars = temp(i,:);
    m1 = model0(pars);
    m1 = calState(m1,N,I);
    m1 = calF(m1);
    plotCnorm(m1,N,I);
    title(tlts{i});
end
ax = gcf;
exportgraphics(ax,'countour_plot_linear_noscale.png');
toc


%% check the objfunc 
pars = [ones(1,1)*.1 ones(1,2)]; 
% pars =[parsSpace(b,:) 0 0 0 1 0 1];
[~,rsqred,resid] = objfunc0(pars,exp_matrix,10,1);
plot(resid,'o-');
ax = gcf;
exportgraphics(ax,'residuals_noscale.png');
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
parsSpace = sort(parsSpace);

%%
% rmsd =cell(ntvec,1); 
% resid=cell(ntvec,1);

rmsd = zeros(numbPoints,1); resid= zeros(numbPoints,7);
% start parallel
tic
ncpu=12;
pc=parcluster('local');
pc.NumThreads=2;
parpool(pc,ncpu)

for j = 1:ntvec
   parfor i = 1:numbPoints
        [rmsd(i),~,resid(i,:)]= objfunc0([parsSpace(i,:) tvec(j,:)],...
            exp_matrix,1,1);
        if mod(i,numbPoints/10)==0
             disp(i)
         end
   end
    disp(j)
    save(['../data/model0_noscale_lin10_normb',num2str(j),'.mat'],...
        'parsSpace','resid','rmsd')
end

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
    load(['../data/model0_noscale_lin10_normb',num2str(jth),'.mat']);
    [minRmsd(jth), minRmsdind(jth)] =   min(rmsd);
    rmsdQuantile{jth} = rmsd(rmsd<prctile(rmsd,10));
    param = parsSpace(minRmsdind(jth),:);
    minRmsdParam(:,jth) = param;
end
%
save('../data/model0_noscale_rmsd_all.mat','minRmsd','minRmsdind','rmsdQuantile', 'minRmsdParam')

tnames = ["t3", "t2", "t1", "t4"];
M = [tnames; minRmsd; minRmsdParam];
save('../data/model0_noscale_minimums.mat', 'M');

%% Plot minimum RMSD and corresponding parameter
figure;
subplot(2,1,1);
bar(minRmsd);
subplot(2,1,2);
bar(minRmsdParam);
ax = gcf;
exportgraphics(ax,'model0_minRMSD_noscale.png');

%% model best-fitting eg : 1.enhanceosome 2. oR 3. IRF. 4. NF
tic
N= linspace(0,1,1000);
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
    plotCnorm(m1,N,I, exp_matrix);
    title(tlts{i})
end
toc
figure;
subplot(2,1,1);
bar(minRmsd);
subplot(2,1,2);
bar(minRmsdParam);
ax = gcf;
exportgraphics(ax,'model0_minRMSD_noscale.png');

% for i =1:4
%     subplot(2,2,i)
%     pars = temp(i,:);
%     m1 = model0(pars);
%     m1 = calState(m1,N,I);
%     m1 = calF(m1);
%     plotCnorm(m1,N,I);
%     title(tlts{i})
% end
