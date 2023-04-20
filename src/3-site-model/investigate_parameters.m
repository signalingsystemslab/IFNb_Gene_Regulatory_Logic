%%  Model B1 experimental parameters
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
load('../data/bestFit_3site_b1.mat')
tts = {'t1: IRF','t2: IRF','t3: NF\kappa B',...
    't4: IRF/IRF','t5: IRF/NF\kappa B','t6: IRF/IRF/NF\kappa B'};
irf_locs=[1,2];
irf_n_locs=[3,4];

% keep same
nbins=50;
edges=zeros(nbins+1,ncpars);
for i=1:ncpars
    edge = linspace(min([parsSpace(:,i);0]), max([parsSpace(:,i);1]),nbins+1);
    edges(:,i)=edge;
end


f=figure;
f.Position(3:4) = [420*3, 420];

for i=1:ncpars
    subplot(2,ncpars,i)
    histogram(parsSpace(:,i),edges(:,i),"FaceColor","black","EdgeColor","none")
    title(tts{i})
    
end


for i=1:ncpars
    subplot(2,ncpars,ncpars+i)
    histogram(parsFinal(:,i),edges(:,i),"FaceColor","black","EdgeColor","none")
    xlabel("Parameter value")
    
end
subplot(2,ncpars,1)
ylabel("Count: randomized")
subplot(2,ncpars,ncpars+1)
ylabel("Count: optimized")

sgtitle(sprintf("%s Experimental Data Parameters",m_name));

ax = gcf;
fig_name = sprintf("../3-site-model/figs/%s_param_hist_exp.png",m_str);
exportgraphics(ax,fig_name);

%  Plot correlation between parameters
g = figure;
g.Position(3:4) = [420*2, 420];
subplot(1,2,1)
[~,ax] = plotmatrix(parsSpace,parsSpace);
title("Randomized parameters")
for i=1:ncpars
    y=ylabel(ax(i,1),tts{i},"Rotation",0);
    y.Position(1)=-2;
%     x=xlabel(ax(ncpars,i),tts{i},"Rotation",45);
%     x.Position(2)=-1;

end
    

subplot(1,2,2)
plotmatrix(parsFinal,parsFinal)
title("Optimized parameters")
% for i=1:ncpars
%     y=ylabel(ax(i,1),tts{i},"Rotation",0);
%     y.Position(1)=-2;
% %     x=xlabel(ax(ncpars,i),tts{i},"Rotation",45);
% %     x.Position(2)=-1;
% 
% end

sgtitle(sprintf("%s Experimental Data Parameters Correlation",m_name));

ax = gcf;
fig_name = sprintf("../3-site-model/figs/%s_param_corr_exp.png",m_str);
exportgraphics(ax,fig_name);

% Export all exp parameters to csv
writematrix(parsFinal,sprintf('../data/Model%s_all_exp_opt_params.csv',m_str))

% Plot parameters but take sum of "same" parameter
ps1 = parsSpace(:,1:irf_locs(1)-1);
psIRF = parsSpace(:,irf_locs(1))+parsSpace(:,irf_locs(2));
ps2 = parsSpace(:,irf_locs(2)+1:irf_n_locs(1)-1);
psIRFn = parsSpace(:,irf_n_locs(1))+parsSpace(:,irf_n_locs(2));
ps3 = parsSpace(:,irf_n_locs(2)+1:ncpars);
ps_new=horzcat(ps1,psIRF,ps2,psIRFn,ps3);

pf1 = parsFinal(:,1:irf_locs(1)-1);
pfIRF = parsFinal(:,irf_locs(1))+parsFinal(:,irf_locs(2));
pf2 = parsFinal(:,irf_locs(2)+1:irf_n_locs(1)-1);
pfIRFn = parsFinal(:,irf_n_locs(1))+parsFinal(:,irf_n_locs(2));
pf3 = parsFinal(:,irf_n_locs(2)+1:ncpars);
pf_new=horzcat(pf1,pfIRF,pf2,pfIRFn,pf3);
tts_sum={tts{1:irf_locs(1)-1},'t1+t2: IRF',tts{irf_locs(2)+1:irf_n_locs(1)-1},'t3+t4: IRF/NF\kappa B',tts{irf_n_locs(2)+1:ncpars}};

nbins=50;
edges=zeros(nbins+1,width(ps_new));
for i=1:width(ps_new)
    edge = linspace(min(ps_new(:,i))-0.01, max(ps_new(:,i))+0.01,nbins+1);
    edges(:,i)=edge;
end


f=figure;
f.Position(3:4) = [420*3, 420];

for i=1:width(ps_new)
    subplot(2,width(ps_new),i)
    histogram(ps_new(:,i),edges(:,i),"FaceColor","black","EdgeColor","none")
    title(tts_sum{i})
    
end

for i=1:width(pf_new)
    subplot(2,width(pf_new),width(pf_new)+i)
    histogram(pf_new(:,i),edges(:,i),"FaceColor","black","EdgeColor","none")
    xlabel("Parameter value")
    
end
subplot(2,width(pf_new),1)
ylabel("Count: randomized")
subplot(2,width(pf_new),width(pf_new)+1)
ylabel("Count: optimized")

sgtitle(sprintf("%s Experimental Data Parameters",m_name));

ax = gcf;
fig_name = sprintf("../3-site-model/figs/%s_param_hist_sums_exp.png",m_str);
exportgraphics(ax,fig_name);
%% Model B1 synthetic parameters

%%  Model B2 experimental parameters
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
tts = {"K_{I2}",'t1: IRF','t2: IRF','t3: NF\kappa B',...
    't4: IRF/IRF','t5: IRF/NF\kappa B','t6: IRF/IRF/NF\kappa B'};
irf_locs=[2,3];
irf_n_locs=[4,5];
% keep same
nbins=50;
edges=zeros(nbins+1,ncpars);
for i=1:ncpars
    edge = linspace(min([parsSpace(:,i);0]), max(parsSpace(:,i))+0.01,nbins+1);
    edges(:,i)=edge;
end


f=figure;
f.Position(3:4) = [420*3, 420];

for i=1:ncpars
    subplot(2,ncpars,i)
    histogram(parsSpace(:,i),edges(:,i),"FaceColor","black","EdgeColor","none")
    title(tts{i})
    
end


for i=1:ncpars
    subplot(2,ncpars,ncpars+i)
    histogram(parsFinal(:,i),edges(:,i),"FaceColor","black","EdgeColor","none")
    xlabel("Parameter value")
    
end
subplot(2,ncpars,1)
ylabel("Count: randomized")
subplot(2,ncpars,ncpars+1)
ylabel("Count: optimized")

sgtitle(sprintf("%s Experimental Data Parameters",m_name));

ax = gcf;
fig_name = sprintf("../3-site-model/figs/%s_param_hist_exp.png",m_str);
ax.PaperPosition = [0 0 12 3];

exportgraphics(ax,fig_name);

%  Plot correlation between parameters
g = figure;
g.Position(3:4) = [420*2, 420];
subplot(1,2,1)
[~,ax] = plotmatrix(parsSpace,parsSpace);
title("Randomized parameters")
for i=1:ncpars
    y=ylabel(ax(i,1),tts{i},"Rotation",0);
    y.Position(1)=-2;
%     x=xlabel(ax(ncpars,i),tts{i},"Rotation",45);
%     x.Position(2)=-1;

end
    

subplot(1,2,2)
plotmatrix(parsFinal,parsFinal)
title("Optimized parameters")
% for i=1:ncpars
%     y=ylabel(ax(i,1),tts{i},"Rotation",0);
%     y.Position(1)=-2;
% %     x=xlabel(ax(ncpars,i),tts{i},"Rotation",45);
% %     x.Position(2)=-1;
% 
% end

sgtitle(sprintf("%s Experimental Data Parameters Correlation",m_name));

ax = gcf;
fig_name = sprintf("../3-site-model/figs/%s_param_corr_exp.png",m_str);
exportgraphics(ax,fig_name);

% Export all exp parameters to csv
writematrix(parsFinal,sprintf('../data/Model%s_all_exp_opt_params.csv',m_str))

% Plot parameters but take sum of "same" parameter
ps1 = parsSpace(:,1:irf_locs(1)-1);
psIRF = parsSpace(:,irf_locs(1))+parsSpace(:,irf_locs(2));
ps2 = parsSpace(:,irf_locs(2)+1:irf_n_locs(1)-1);
psIRFn = parsSpace(:,irf_n_locs(1))+parsSpace(:,irf_n_locs(2));
ps3 = parsSpace(:,irf_n_locs(2)+1:ncpars);
ps_new=horzcat(ps1,psIRF,ps2,psIRFn,ps3);

pf1 = parsFinal(:,1:irf_locs(1)-1);
pfIRF = parsFinal(:,irf_locs(1))+parsFinal(:,irf_locs(2));
pf2 = parsFinal(:,irf_locs(2)+1:irf_n_locs(1)-1);
pfIRFn = parsFinal(:,irf_n_locs(1))+parsFinal(:,irf_n_locs(2));
pf3 = parsFinal(:,irf_n_locs(2)+1:ncpars);
pf_new=horzcat(pf1,pfIRF,pf2,pfIRFn,pf3);
tts_sum={tts{1:irf_locs(1)-1},'t1+t2: IRF',tts{irf_locs(2)+1:irf_n_locs(1)-1},'t3+t4: IRF/NF\kappa B',tts{irf_n_locs(2)+1:ncpars}};

nbins=50;
edges=zeros(nbins+1,width(ps_new));
for i=1:width(ps_new)
    edge = linspace(min(ps_new(:,i))-0.01, max(ps_new(:,i))+0.01,nbins+1);
    edges(:,i)=edge;
end


f=figure;
f.Position(3:4) = [420*3, 420];

for i=1:width(ps_new)
    subplot(2,width(ps_new),i)
    histogram(ps_new(:,i),edges(:,i),"FaceColor","black","EdgeColor","none")
    title(tts_sum{i})
    
end

for i=1:width(pf_new)
    subplot(2,width(pf_new),width(pf_new)+i)
    histogram(pf_new(:,i),edges(:,i),"FaceColor","black","EdgeColor","none")
    xlabel("Parameter value")
    
end
subplot(2,width(pf_new),1)
ylabel("Count: randomized")
subplot(2,width(pf_new),width(pf_new)+1)
ylabel("Count: optimized")

sgtitle(sprintf("%s Experimental Data Parameters",m_name));

ax = gcf;
fig_name = sprintf("../3-site-model/figs/%s_param_hist_sums_exp.png",m_str);
exportgraphics(ax,fig_name);
%%  Model B3 experimental parameters
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
tts = {"C",'t1: IRF','t2: IRF','t3: NF\kappa B',...
    't4: IRF/IRF','t5: IRF/NF\kappa B','t6: IRF/IRF/NF\kappa B'};
irf_locs=[2,3];
irf_n_locs=[4,5];

% keep same
nbins=50;
edges=zeros(nbins+1,ncpars);
for i=1:ncpars
    edge = linspace(min(parsSpace(:,i)-.01), max(parsSpace(:,i))+0.01,nbins+1);
    edges(:,i)=edge;
end


f=figure;
f.Position(3:4) = [420*3, 420];

for i=1:ncpars
    subplot(2,ncpars,i)
    histogram(parsSpace(:,i),edges(:,i),"FaceColor","black","EdgeColor","none")
    title(tts{i})
    
end


for i=1:ncpars
    subplot(2,ncpars,ncpars+i)
    histogram(parsFinal(:,i),edges(:,i),"FaceColor","black","EdgeColor","none")
    xlabel("Parameter value")
    
end
subplot(2,ncpars,1)
ylabel("Count: randomized")
subplot(2,ncpars,ncpars+1)
ylabel("Count: optimized")

sgtitle(sprintf("%s Experimental Data Parameters",m_name));

ax = gcf;
fig_name = sprintf("../3-site-model/figs/%s_param_hist_exp.png",m_str);
ax.PaperPosition = [0 0 12 3];

exportgraphics(ax,fig_name);

%  Plot correlation between parameters
g = figure;
g.Position(3:4) = [420*2, 420];
subplot(1,2,1)
[~,ax] = plotmatrix(parsSpace,parsSpace);
title("Randomized parameters")
for i=1:ncpars
    y=ylabel(ax(i,1),tts{i},"Rotation",0);
    y.Position(1)=-2;
%     x=xlabel(ax(ncpars,i),tts{i},"Rotation",45);
%     x.Position(2)=-1;

end
    

subplot(1,2,2)
plotmatrix(parsFinal,parsFinal)
title("Optimized parameters")
% for i=1:ncpars
%     y=ylabel(ax(i,1),tts{i},"Rotation",0);
%     y.Position(1)=-2;
% %     x=xlabel(ax(ncpars,i),tts{i},"Rotation",45);
% %     x.Position(2)=-1;
% 
% end

sgtitle(sprintf("%s Experimental Data Parameters Correlation",m_name));

ax = gcf;
fig_name = sprintf("../3-site-model/figs/%s_param_corr_exp.png",m_str);
exportgraphics(ax,fig_name);

% Export all exp parameters to csv
writematrix(parsFinal,sprintf('../data/Model%s_all_exp_opt_params.csv',m_str))

% Plot parameters but take sum of "same" parameter
ps1 = parsSpace(:,1:irf_locs(1)-1);
psIRF = parsSpace(:,irf_locs(1))+parsSpace(:,irf_locs(2));
ps2 = parsSpace(:,irf_locs(2)+1:irf_n_locs(1)-1);
psIRFn = parsSpace(:,irf_n_locs(1))+parsSpace(:,irf_n_locs(2));
ps3 = parsSpace(:,irf_n_locs(2)+1:ncpars);
ps_new=horzcat(ps1,psIRF,ps2,psIRFn,ps3);

pf1 = parsFinal(:,1:irf_locs(1)-1);
pfIRF = parsFinal(:,irf_locs(1))+parsFinal(:,irf_locs(2));
pf2 = parsFinal(:,irf_locs(2)+1:irf_n_locs(1)-1);
pfIRFn = parsFinal(:,irf_n_locs(1))+parsFinal(:,irf_n_locs(2));
pf3 = parsFinal(:,irf_n_locs(2)+1:ncpars);
pf_new=horzcat(pf1,pfIRF,pf2,pfIRFn,pf3);
tts_sum={tts{1:irf_locs(1)-1},'t1+t2: IRF',tts{irf_locs(2)+1:irf_n_locs(1)-1},'t3+t4: IRF/NF\kappa B',tts{irf_n_locs(2)+1:ncpars}};

nbins=50;
edges=zeros(nbins+1,width(ps_new));
for i=1:width(ps_new)
    edge = linspace(min(ps_new(:,i))-0.01, max(ps_new(:,i))+0.01,nbins+1);
    edges(:,i)=edge;
end


f=figure;
f.Position(3:4) = [420*3, 420];

for i=1:width(ps_new)
    subplot(2,width(ps_new),i)
    histogram(ps_new(:,i),edges(:,i),"FaceColor","black","EdgeColor","none")
    title(tts_sum{i})
    
end

for i=1:width(pf_new)
    subplot(2,width(pf_new),width(pf_new)+i)
    histogram(pf_new(:,i),edges(:,i),"FaceColor","black","EdgeColor","none")
    xlabel("Parameter value")
    
end
subplot(2,width(pf_new),1)
ylabel("Count: randomized")
subplot(2,width(pf_new),width(pf_new)+1)
ylabel("Count: optimized")

sgtitle(sprintf("%s Experimental Data Parameters",m_name));

ax = gcf;
fig_name = sprintf("../3-site-model/figs/%s_param_hist_sums_exp.png",m_str);
exportgraphics(ax,fig_name);
%%  Model B4 experimental parameters
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
parsSpace(:,2) = ((parsSpace(:,2)-0.5)*4);% -2 to 2 for C
m_name = "Model \beta 4";
m_str="B4";
load('../data/bestFit_3site_b4.mat')
tts = {"K_{I2}","C",'t1: IRF','t2: IRF','t3: NF\kappa B',...
    't4: IRF/IRF','t5: IRF/NF\kappa B','t6: IRF/IRF/NF\kappa B'};
irf_locs=[3,4];
irf_n_locs=[5,6];

% keep same
nbins=50;
edges=zeros(nbins+1,ncpars);
for i=1:ncpars
    edge = linspace(min(parsSpace(:,i)-.01), max(parsSpace(:,i))+0.01,nbins+1);
    edges(:,i)=edge;
end


f=figure;
f.Position(3:4) = [420*3, 420];

for i=1:ncpars
    subplot(2,ncpars,i)
    histogram(parsSpace(:,i),edges(:,i),"FaceColor","black","EdgeColor","none")
    title(tts{i})
    
end


for i=1:ncpars
    subplot(2,ncpars,ncpars+i)
    histogram(parsFinal(:,i),edges(:,i),"FaceColor","black","EdgeColor","none")
    xlabel("Parameter value")
    
end
subplot(2,ncpars,1)
ylabel("Count: randomized")
subplot(2,ncpars,ncpars+1)
ylabel("Count: optimized")

sgtitle(sprintf("%s Experimental Data Parameters",m_name));

ax = gcf;
fig_name = sprintf("../3-site-model/figs/%s_param_hist_exp.png",m_str);
ax.PaperPosition = [0 0 12 3];

exportgraphics(ax,fig_name);

%  Plot correlation between parameters
g = figure;
g.Position(3:4) = [420*3, 420];
subplot(1,2,1)
[~,ax] = plotmatrix(parsSpace,parsSpace);
title("Randomized parameters")
for i=1:ncpars
    y=ylabel(ax(i,1),tts{i},"Rotation",0);
    y.Position(1)=-3;
%     x=xlabel(ax(ncpars,i),tts{i},"Rotation",45);
%     x.Position(2)=-1;

end
    

subplot(1,2,2)
plotmatrix(parsFinal,parsFinal)
title("Optimized parameters")
% for i=1:ncpars
%     y=ylabel(ax(i,1),tts{i},"Rotation",0);
%     y.Position(1)=-2;
% %     x=xlabel(ax(ncpars,i),tts{i},"Rotation",45);
% %     x.Position(2)=-1;
% 
% end

sgtitle(sprintf("%s Experimental Data Parameters Correlation",m_name));

ax = gcf;
fig_name = sprintf("../3-site-model/figs/%s_param_corr_exp.png",m_str);
exportgraphics(ax,fig_name);

% Export all exp parameters to csv
writematrix(parsFinal,sprintf('../data/Model%s_all_exp_opt_params.csv',m_str))
% Plot parameters but take sum of "same" parameter
ps1 = parsSpace(:,1:irf_locs(1)-1);
psIRF = parsSpace(:,irf_locs(1))+parsSpace(:,irf_locs(2));
ps2 = parsSpace(:,irf_locs(2)+1:irf_n_locs(1)-1);
psIRFn = parsSpace(:,irf_n_locs(1))+parsSpace(:,irf_n_locs(2));
ps3 = parsSpace(:,irf_n_locs(2)+1:ncpars);
ps_new=horzcat(ps1,psIRF,ps2,psIRFn,ps3);

pf1 = parsFinal(:,1:irf_locs(1)-1);
pfIRF = parsFinal(:,irf_locs(1))+parsFinal(:,irf_locs(2));
pf2 = parsFinal(:,irf_locs(2)+1:irf_n_locs(1)-1);
pfIRFn = parsFinal(:,irf_n_locs(1))+parsFinal(:,irf_n_locs(2));
pf3 = parsFinal(:,irf_n_locs(2)+1:ncpars);
pf_new=horzcat(pf1,pfIRF,pf2,pfIRFn,pf3);
tts_sum={tts{1:irf_locs(1)-1},'t1+t2: IRF',tts{irf_locs(2)+1:irf_n_locs(1)-1},'t3+t4: IRF/NF\kappa B',tts{irf_n_locs(2)+1:ncpars}};

nbins=50;
edges=zeros(nbins+1,width(ps_new));
for i=1:width(ps_new)
    edge = linspace(min(ps_new(:,i))-0.01, max(ps_new(:,i))+0.01,nbins+1);
    edges(:,i)=edge;
end


f=figure;
f.Position(3:4) = [420*3, 420];

for i=1:width(ps_new)
    subplot(2,width(ps_new),i)
    histogram(ps_new(:,i),edges(:,i),"FaceColor","black","EdgeColor","none")
    title(tts_sum{i})
    
end

for i=1:width(pf_new)
    subplot(2,width(pf_new),width(pf_new)+i)
    histogram(pf_new(:,i),edges(:,i),"FaceColor","black","EdgeColor","none")
    xlabel("Parameter value")
    
end
subplot(2,width(pf_new),1)
ylabel("Count: randomized")
subplot(2,width(pf_new),width(pf_new)+1)
ylabel("Count: optimized")

sgtitle(sprintf("%s Experimental Data Parameters",m_name));

ax = gcf;
fig_name = sprintf("../3-site-model/figs/%s_param_hist_sums_exp.png",m_str);
exportgraphics(ax,fig_name);
