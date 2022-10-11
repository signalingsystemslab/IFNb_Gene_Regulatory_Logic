
    loadsetting;

%% fig1.exp_tb: experimental data table 

fig1.exp_tb = figure('PaperPositionMode','auto','unit','inches','position',[  20   20  4.5*2  3.6]); hold on; 

tmp = [exp_matrix.irf; exp_matrix.nfkb; exp_matrix.ifnb];
tmp = [tmp(:,1:3) ones(3,1)*2,tmp(:,4:7)];
plot_data = [tmp(:,1:4)' tmp(:,5:end)'];
%
% cmp=[ colormap(redbluecmap(10,'interpolation','linear')); 0 0 0;];
% cmp=[flipud(othercolor('Greens8',10)); ;];
cmp = [colorGradient([1 1 1],[1 0 0],21);0.7 0.7 0.7];
% colormap(parula(8))
im = imagesc(plot_data,[-0.001,1.05]); colormap(cmp)%h = colorbar;
% im.AlphaData = 0.5;
% image(plot_data,'CDataMapping','scaled')
set(gca,'yDir','reverse')
% axis off
caxis([0 ,1.2])
axis([.5 6.5 .5 4.5]), axis off;
hold on ; 
yax = 4; xax =6; xaxs = 1:(xax+1);%-0.5;
yaxs = 1:(yax+1);%-0.5;
arrayfun(@(x) plot([x x]-.5,[0 yaxs(end)]+.5,'k'),xaxs)%(1:end-1))
arrayfun(@(y) plot([0 xaxs(end)]+.5,[y y]-.5,'k'),yaxs)%(1:end-1))

