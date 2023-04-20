%% Model B1 residuals
% Calculate residuals from random parameters
clear;

load ../data/exp_matrix_norm.mat

seed = 5;
rng(seed) % set seed for rands
npts = 99; % number of synthetic points
nexp = length(exp_matrix.irf); % number of points in an experiment
numbPoints = 10^4; % number of vectors

% Model specific
ncpars = 6; 
parsSpace = rand(ncpars*numbPoints,1); 
parsSpace= reshape(parsSpace,numbPoints,ncpars);
m_name = "Model \beta 1";
m_str="B1";


tts = {'pt1','pt2','pt3','pt4','pt5','pt6','pt7'};
rsq_rand = zeros(numbPoints,1); resid_rand= zeros(numbPoints,nexp); rmsd_rand = zeros(numbPoints,1);
% 
% %% 
% % start parallel pool
% ncpu=20;
% pc=parcluster('local');
% pc.NumThreads=2;
% parpool(pc,ncpu);
% 
% tic
% fprintf("Starting optimization\n")
% parfor i = 1:numbPoints
% % for i = 1:5
% %   Find local minima (of residuals) near each random parameter set
% %     Optimize a vector by finding minimum of sum of squares (rmsd)
%     [rmsd_rand(i,:),rsq_rand(i,:),resid_rand(i,:)]= objfuncB1(parsSpace(i,:), exp_matrix,1,1);
%     if mod(i,numbPoints/10)==0
%         disp(i)
%     end
% end
% 
% % Calculate AIC and RMSD
% aic_rand = nexp*log(rmsd_rand.^2)+2*ncpars;
% 
% 
% save('../data/randomFit_3site_b1.mat','parsSpace','resid_rand','rmsd_rand', ...
%     "aic_rand")
%%
load('../data/randomFit_3site_b1.mat')
load('../data/bestFit_3site_b1.mat')
nbins=30;
edges=zeros(nbins+1,nexp);
for i=1:nexp
    edge = linspace(min(resid(:,i)-0.01), max(resid(:,i)+0.01),nbins+1);
    edges(:,i)=edge;
end

edges2=zeros(nbins+1,nexp);
for i=1:nexp
    edge = linspace(0,1,nbins+1);
    edges2(:,i)=edge;
end

f=figure;
f.Position(3:4) = [420*3, 420];

for i=1:nexp
    subplot(2,nexp,i)
    histogram(resid(:,i),edges(:,i),"FaceColor","black","EdgeColor","none")
    title(tts{i})
end

for i=1:nexp
    subplot(2,nexp,nexp+i)
    histogram(resid_rand(:,i),edges2(:,i),"FaceColor","black","EdgeColor","none")
end

subplot(2,nexp,nexp+1)
title("Random Parameters")
sgtitle(sprintf("Model %s Distribution of residuals from optimized parameters and random parameters",m_str))

ax = gcf;
fig_name = sprintf("../3-site-model/figs/%s_resid_hist_exp.png",m_str);
exportgraphics(ax,fig_name);

%% 



%% Model B2 residuals
% Calculate residuals from random parameters
clear;

load ../data/exp_matrix_norm.mat

seed = 5;
rng(seed) % set seed for rands
npts = 99; % number of synthetic points
nexp = length(exp_matrix.irf); % number of points in an experiment
numbPoints = 10^4; % number of vectors

% Model specific
ncpars = 7;
parsSpace = rand(ncpars*numbPoints,1); 
parsSpace= reshape(parsSpace,numbPoints,ncpars);
parsSpace(:,1) = ((parsSpace(:,1))*2); % 0 to 2 for K_I2 allows 2-fold difference in IRF binding
m_name = "Model \beta 2";
m_str="B2";
load('../data/bestFit_3site_b2.mat')

tts = {'pt1','pt2','pt3','pt4','pt5','pt6','pt7'};
rsq_rand = zeros(numbPoints,1); resid_rand= zeros(numbPoints,nexp); rmsd_rand = zeros(numbPoints,1);

% %% 
% tic
% fprintf("Starting optimization\n")
% parfor i = 1:numbPoints
% % for i = 1:5
% %   Find local minima (of residuals) near each random parameter set
% %     Optimize a vector by finding minimum of sum of squares (rmsd)
%     [rmsd_rand(i,:),rsq_rand(i,:),resid_rand(i,:)]= objfuncB2(parsSpace(i,:), exp_matrix,1,1);
%     if mod(i,numbPoints/10)==0
%         disp(i)
%     end
% end
% 
% % Calculate AIC and RMSD
% aic_rand = nexp*log(rmsd_rand.^2)+2*ncpars;
% 
% 
% save(sprintf('../data/randomFit_3site_%s.mat',m_str),'parsSpace','resid_rand','rmsd_rand', ...
%     "aic_rand")
%%
load(sprintf('../data/randomFit_3site_%s.mat',m_str))
nbins=30;
edges=zeros(nbins+1,nexp);
for i=1:nexp
    edge = linspace(min(resid(:,i)-0.01), max(resid(:,i)+0.01),nbins+1);
    edges(:,i)=edge;
end

edges2=zeros(nbins+1,nexp);
for i=1:nexp
    edge = linspace(0,1,nbins+1);
    edges2(:,i)=edge;
end

f=figure;
f.Position(3:4) = [420*3, 420];

for i=1:nexp
    subplot(2,nexp,i)
    histogram(resid(:,i),edges(:,i),"FaceColor","black","EdgeColor","none")
    title(tts{i})
end

for i=1:nexp
    subplot(2,nexp,nexp+i)
    histogram(resid_rand(:,i),edges2(:,i),"FaceColor","black","EdgeColor","none")
end

subplot(2,nexp,nexp+1)
title("Random Parameters")
sgtitle(sprintf("Model %s Distribution of residuals from optimized parameters and random parameters",m_str))

ax = gcf;
fig_name = sprintf("../3-site-model/figs/%s_resid_hist_exp.png",m_str);
exportgraphics(ax,fig_name);
%%



%% Model B3 residuals
% Calculate residuals from random parameters
clear;

load ../data/exp_matrix_norm.mat

seed = 5;
rng(seed) % set seed for rands
npts = 99; % number of synthetic points
nexp = length(exp_matrix.irf); % number of points in an experiment
numbPoints = 10^4; % number of vectors

% Model specific
ncpars = 7;
parsSpace = rand(ncpars*numbPoints,1); 
parsSpace= reshape(parsSpace,numbPoints,ncpars);
% parsSpace(:,1) = (parsSpace(:,1)*4); % 0 to 4
parsSpace(:,1) = ((parsSpace(:,1)-0.5)*4);% -2 to 2 for C parameter
m_name = "Model \beta 3";
m_str="B3";
load('../data/bestFit_3site_b3.mat')

tts = {'pt1','pt2','pt3','pt4','pt5','pt6','pt7'};
rsq_rand = zeros(numbPoints,1); resid_rand= zeros(numbPoints,nexp); rmsd_rand = zeros(numbPoints,1);

% %% 
% % start parallel pool
% tic
% fprintf("Starting optimization\n")
% parfor i = 1:numbPoints
% % for i = 1:5
% %   Find local minima (of residuals) near each random parameter set
% %     Optimize a vector by finding minimum of sum of squares (rmsd)
%     [rmsd_rand(i,:),rsq_rand(i,:),resid_rand(i,:)]= objfuncB3(parsSpace(i,:), exp_matrix,1,1);
%     if mod(i,numbPoints/10)==0
%         disp(i)
%     end
% end
% 
% % Calculate AIC and RMSD
% aic_rand = nexp*log(rmsd_rand.^2)+2*ncpars;
% 
% 
% save(sprintf('../data/randomFit_3site_%s.mat',m_str),'parsSpace','resid_rand','rmsd_rand', ...
%     "aic_rand")
%%
load(sprintf('../data/randomFit_3site_%s.mat',m_str))
nbins=30;
edges=zeros(nbins+1,nexp);
for i=1:nexp
    edge = linspace(min(resid(:,i)-0.01), max(resid(:,i)+0.01),nbins+1);
    edges(:,i)=edge;
end

edges2=zeros(nbins+1,nexp);
for i=1:nexp
    edge = linspace(0,1,nbins+1);
    edges2(:,i)=edge;
end

f=figure;
f.Position(3:4) = [420*3, 420];

for i=1:nexp
    subplot(2,nexp,i)
    histogram(resid(:,i),edges(:,i),"FaceColor","black","EdgeColor","none")
    title(tts{i})
end

for i=1:nexp
    subplot(2,nexp,nexp+i)
    histogram(resid_rand(:,i),edges2(:,i),"FaceColor","black","EdgeColor","none")
end

subplot(2,nexp,nexp+1)
title("Random Parameters")
sgtitle(sprintf("Model %s Distribution of residuals from optimized parameters and random parameters",m_str))

ax = gcf;
fig_name = sprintf("../3-site-model/figs/%s_resid_hist_exp.png",m_str);
exportgraphics(ax,fig_name);
%%



%% Model B4 residuals
% Calculate residuals from random parameters
clear;

load ../data/exp_matrix_norm.mat

seed = 5;
rng(seed) % set seed for rands
npts = 99; % number of synthetic points
nexp = length(exp_matrix.irf); % number of points in an experiment
numbPoints = 10^4; % number of vectors

% Model specific
ncpars = 8;
parsSpace = rand(ncpars*numbPoints,1); 
parsSpace= reshape(parsSpace,numbPoints,ncpars);
parsSpace(:,1) = (parsSpace(:,1)*2); % 0 to 2
parsSpace(:,2) = (parsSpace(:,2)*2);% 0 to 2
m_name = "Model \beta 4";
m_str="B4";
load('../data/bestFit_3site_b4.mat')

tts = {'pt1','pt2','pt3','pt4','pt5','pt6','pt7'};
rsq_rand = zeros(numbPoints,1); resid_rand= zeros(numbPoints,nexp); rmsd_rand = zeros(numbPoints,1);

% %% 
% % start parallel pool
% 
% tic
% fprintf("Starting optimization\n")
% parfor i = 1:numbPoints
% % for i = 1:5
% %   Find local minima (of residuals) near each random parameter set
% %     Optimize a vector by finding minimum of sum of squares (rmsd)
%     [rmsd_rand(i,:),rsq_rand(i,:),resid_rand(i,:)]= objfuncB4(parsSpace(i,:), exp_matrix,1,1);
%     if mod(i,numbPoints/10)==0
%         disp(i)
%     end
% end
% 
% % Calculate AIC and RMSD
% aic_rand = nexp*log(rmsd_rand.^2)+2*ncpars;
% 
% 
% save(sprintf('../data/randomFit_3site_%s.mat',m_str),'parsSpace','resid_rand','rmsd_rand', ...
%     "aic_rand")
%%
load(sprintf('../data/randomFit_3site_%s.mat',m_str))
nbins=30;
edges=zeros(nbins+1,nexp);
for i=1:nexp
    edge = linspace(min(resid(:,i)-0.01), max(resid(:,i)+0.01),nbins+1);
    edges(:,i)=edge;
end

edges2=zeros(nbins+1,nexp);
for i=1:nexp
    edge = linspace(0,1,nbins+1);
    edges2(:,i)=edge;
end

f=figure;
f.Position(3:4) = [420*3, 420];

for i=1:nexp
    subplot(2,nexp,i)
    histogram(resid(:,i),edges(:,i),"FaceColor","black","EdgeColor","none")
    title(tts{i})
end

for i=1:nexp
    subplot(2,nexp,nexp+i)
    histogram(resid_rand(:,i),edges2(:,i),"FaceColor","black","EdgeColor","none")
end

subplot(2,nexp,nexp+1)
title("Random Parameters")
sgtitle(sprintf("Model %s Distribution of residuals from optimized parameters and random parameters",m_str))

ax = gcf;
fig_name = sprintf("../3-site-model/figs/%s_resid_hist_exp.png",m_str);
exportgraphics(ax,fig_name);