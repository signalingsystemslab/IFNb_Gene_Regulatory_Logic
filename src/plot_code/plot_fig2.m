clear; loadsetting
%% a. fig2.exp_hm: bar_min_score
load ../data/model0_rmsd_all.mat
stdv = cellfun(@std,rmsdQuantile);
meanv = cellfun(@mean,rmsdQuantile);

if ~ exist('fig2') 
    fig2.bar_min_score = figure('Unit','inches','PaperPositionMode','auto','Position',[5 5 2 2]);
else ~isfield(fig2,'bar_min_score')
    fig2.bar_min_score = figure('Unit','inches','PaperPositionMode','auto','Position',[5 5 2 2]);
end
barwitherr(stdv,meanv,'facecolor',[127 255 0]/255)
xlim([.5 4.5])
set(gca,'xticklabel',{},'yticklabel',{})
%% c. fig2.exp_hm:experimental data figure 
% visualize expdata 

if numel(exp_matrix.irf) ==8
    exp_matrix.irf(7) =0.1;
    exp_matrix.irf(6) =[];exp_matrix.nfkb(6) =[];exp_matrix.ifnb(6)=[];
end

fig2.exp_hm = figure('PaperPositionMode','auto','unit','inches','position',[  20   20  4.5  3.6]); hold on; 
cols = flipud(othercolor('RdYlBu_11b'));
plot([exp_matrix.irf([6 1]),exp_matrix.irf(2)+0.2],exp_matrix.nfkb([6 1 2]),...
    'color',cols(round(exp_matrix.ifnb(1)*63)+1,:),'linewidth',5)

arrayfun(@(x) plot(exp_matrix.irf(x),exp_matrix.nfkb(x),'o','markersize',12,...
    'MarkerFaceColor',cols(round(exp_matrix.ifnb(x)*63)+1,:),...
    'markerEdgeColor',[.5 .5 .5],'linewidth',1.5),[1:2 4:7 3])
xlim([-0.1 1.1]);ylim([-0.1 1.1]); set(gca,'xticklabel',{},'yticklabel','')
hold on ;box on; 
% arrayfun(@(x)(text(exp_matrix.irf(x),exp_matrix.nfkb(x)+0.04, num2str(x),...
%          'FontSize',8,'color','k',...
%          'HorizontalAlignment','center')),1:7);
h = colorbar;
set(h,'ytick',0:13:64,'yticklabel',{})
% title(h,'[IFNb]')
% xlabel('[IRF]');ylabel('[NF\kappaB]');save2pdf('exp_hm')
%% b. fig2.heatmps:the heatmap transition
addpath(model_2site)
vs = [0.5 1  2];
tils = {'Negative Cooperativity','No Cooperativity','Positive Cooperativity'};
for i=1:length(vs)
    fig2.heatmp{i}= figure('PaperPositionMode','auto','position',[  840   433   400/2  300/2]); hold on; 
    pars = [vs(i) 1 0];
    N= linspace(0,10,1000);
    I = N;
    cm = colormap(flipud(othercolor('RdYlBu_11b',8)));
    m = model0(pars);
    m = calState(m,N,I);
    m = calF(m);
    plotCnorm(m,N,I);xlim([0 4]);ylim([0 4]);xlabel('');ylabel(''); 
    colorbar off; set(gca,'xticklabel','','yticklabel','');%title(tils{i});
end


