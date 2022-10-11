clear; 
loadsetting;
fit1='_b';

%% fig3B
rmsdAll = zeros(10000,4);
resdiueAll = zeros(10000,7,4); 
for i = 0:3
    load(['../data/bestFit_model',num2str(i),fit1,'.mat'])
    rmsdAll(:,i+1) = rmsd;
    residAll(:,:,i+1)= resid; 
end
% boxplot([rmsd0 rmsd1])
fig3.minRmsd = figure('PaperPositionMode','auto','position',[1600 40 200 200]);
[minRmsd,minRmsdInd] = min(rmsdAll);
bar_h= bar(minRmsd);
bar_child=get(bar_h,'Children');
axis([0 5 0 .15]),set(gca,'xticklabel','','yticklabel','')

set(bar_child,'CDataMapping','direct');
index = [1 2 3 4];
set(bar_child, 'CData',index);
colormap(mycolor);

%% fig3C
fig3.minResid = figure('PaperPositionMode','auto','position',[1600 40 200 200]);
minResid = zeros(4,7);
hold on ;
axis([0 8 -0.2 .3]);
for i = 1:4
    minResid(i,:) = squeeze(residAll(minRmsdInd(i),:,i));
    plot(minResid(i,:),'o','color',mycolor(i,:),'linewidth',1.5); 
end
plot([0 8],[0 0],'--','color',[.1 .1 .1]);hold on; 
plot([0 8],[-.1 -.1],'--','color',[.1 .1 .1]);
plot([0 8],[.1 .1],'--','color',[.1 .1 .1])
set(gca,'xtick',1:7,'xticklabel','','yticklabel',''); box on ;
%legend({'','','',''},'Location','north')
%legend boxoff 

%% fig3D
% th1 = 0.9, th2 =.95
 load ../data/exp_matrix_norm.mat
randx = @(x,nrow) (rand(nrow,1)-0.5)*.3 + x ;
nplot = [1 1 2];

% axis([.5 4.5 -1.1 .5]);hold on; 
for i = 1:3
    eval(['fig3.bestp_AC_box',num2str(i),'=figure(''PaperPositionMode'',''auto'',''position'',[1600 40 ', num2str(75*nplot(i)),' 200])']);
    ha= tight_subplot(1,1)
    load(['../data/bestFit_model',num2str(i),fit1,'.mat'])
    disp(['Model',num2str(i),'''s unique parameters sets:',num2str(size(unique(parsFinal,'rows')))]);
    rsq0  = rsquaredFunc(resid);
    p1 = (rsq0>0.95);
    if i==3
        plot([randx(1,sum(p1)) randx(1.5,sum(p1))],parsFinal(p1,1:2),'o','markersize',2,'color',mycolor(i+1,:))
        set(gca,'xtick',[1 1.5]);axis([0.75 1.75 -1.1 .5])
    else 
        plot(randx(nplot(i),sum(p1)) ,parsFinal(p1,1),'o','markersize',2,'color',mycolor(i+1,:))
        set(gca,'xtick',1);axis([0.75 1.25 -1.1 .5]);
    end
    hold on; plot([0 3],[0 0],'--','color',color.wt,'linewidth',1.5);   
    set(gca,'xticklabel','','yticklabel','');
end

box on ;


%% fig3(E)plot best parameter distributions for t vectors

for i = 1:1
    load(['../data/bestFit_model',num2str(i),fit1,'.mat'])
    rsq0  = rsquaredFunc(resid);
    if i == 0
        p1 = (rsq0>=0.9);
    else
        p1 =(rsq0>=0.95);
    end
    mn =mean(parsFinal(p1,:));
    se = std(parsFinal(p1,:));
    fig3.bestp_t{i+1}=figure('PaperPositionMode','auto','position',[1600 40 200 200]);
    if strcmp(fit1,'_a')
        barwitherr(se,mn,'facecolor',mycolor(i+1,:))
    else
        barwitherr(se(end-5:end),mn(end-5:end),'facecolor',mycolor(i+1,:))
    end
    
    xlim([0.5 6+0.5]);ylim([0 1.2])
    set(gca,'xticklabel','','ytick',0:1,'yticklabel','');
    
end

