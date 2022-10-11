clear
loadsetting;

fit1='_b';
%% load data & simulate p50
im = 1; % model 1
addpath(model_3site);
load(['../data/bestFit_model',num2str(im),fit1,'.mat'])

[v,ind] = min(rmsd); % minimal rmsd

% p50 dependence sim
% relationship: C = K_gIRE/K_IRE = 1 + p50/K_p50_IRE;
cfunc = @(p50) 1./(1+ p50);
cbest = 10.^parsFinal(ind,1);
p50_wt = 1/cbest-1;

p50folds = [1./(10:-.25:1), 1.25:.25:10];pdim = length(p50folds);
pars = [cfunc(p50folds*p50_wt)' repmat(parsFinal(ind,2:end),pdim,1)];
ndim = 1000;
Nf = 0; % 0 for LPS; 1 for RLR 
Irf = linspace(0,10,ndim);
fmat = zeros(ndim,pdim); % col: p50, row: irf

for i = 1:pdim
    eval(['m1 = model',num2str(im),'(pars(i,:));']);
    m1 = calState2(m1,Nf,Irf);
    m1 = calF2(m1);
    fmat(:,i) = m1.f; % col: p50 level
end

%% plot fig5A
normalize = @(x) x/max(x(:));
fig5.p50_predict= figure('Position',[ 680   558   140  105],'paperpositionmode','auto');
tight_subplot(1,1,[.01 .01],[.01 .01],[.01 .01])
fill([1.05 21 21 1.05],[.003 .003 .495 .495],[238 221 130]/255,'edgecolor','none'); 
hold on; 
hf = fill([53 pdim pdim 53],[.003 .003 .495 .495],[255 228 181]/255,'edgecolor','none'); 
plot(fmat(250,:),'k','linewidth',3)
plot([(pdim+1)/2 (pdim+1)/2],[0 .5],'--','color',[255 205 205]/255,'linewidth',1.5)
axis([1 pdim+.5 0 .5])
box on ;
set(gca,'xtick',1:12:pdim,'xticklabel','','yticklabel','')
%% CHen's 3T3 data  (Fig5B) 
%          wt    acko     abcko    ac50 
plotdata =[1	0.725961109	1.140387724	1.724854449; 
10.59635607	1.043739902	5.444330239	7.809227082; 
115.3571651	24.54555445	4.986978287	39.01450433;
44.32974379	7.020704573	2.760974157	5.325367205];

set(0,'defaultAxesColorOrder',[color.ac;color.abc;color.ac50])
fig5.chen3t3_p = figure('Position',[ 680   558   140  105],'paperpositionmode','auto');
tight_subplot(1,1,[.01 .01],[.01 .01],[.01 .01])
plot([0 1 3 6] , plotdata(:,2:end),'o-','linewidth',2) 
axis([-0.2 6.2 -5 70])
set(gca,'xtick',[0 1 3 6],'ytick',0:25:50,'xticklabel','','yticklabel','')
