clear
loadsetting;

fit1='_b';
%% fig4A
% load data 
i = 1; load(['../data/bestFit_model',num2str(i),fit1,'.mat']); 
[v,ind] = min(rmsd); 
Ndim = 1000; 


% parameters 
pars = [10.^parsFinal(ind,1) parsFinal(ind,2:end)];
Irf = linspace(0,10,Ndim);Nf = Irf;
addpath(model_3site);
m1 = model1(pars);
m1 = calState(m1,Nf,Irf);
m1 = calF(m1);

m1.f = m1.f/max(m1.f(:));

% plot
fig4.hm = figure('PaperPositionMode','auto','position',[1600 40 500 400]);
hb = tight_subplot(1,1,[.01 .01],[.01 .01],[.01 .01]);
axis(hb)

imagesc(Nf,Irf,m1.f);set(gca,'xticklabel','','yticklabel','')
h=colorbar;
set(h,'yticklabel','')
set(gca,'ydir','normal')
hold on; 

%% fig4B
fig4.best_rg = figure('PaperPositionMode','auto','position',[800 40 500 200]);
reg_amp.irf = m1.f(:,end)-m1.f(:,1);
reg_amp.nfkb = m1.f(end,:)-m1.f(1,:);
hb = tight_subplot(1,2,[.01 .02],[.01 .01],[.01 .01]);
axes(hb(1))

% plot(Irf,reg_amp.irf,'linewidth',2); 
x=[Irf, fliplr(Irf)]; y= [m1.f(:,1) ;m1.f(end:-1:1,end)]*1000+0.0001;
fill(x,y,ones(1,3)*.9)
axis([0.01 10 1 2*10^3]);
 set(gca,'xticklabel','','yticklabel','')
set(gca,'Layer','top','yscale','log','ytick',10.^(0:4))

axes(hb(2))

% plot(Nf,reg_amp.nfkb,'linewidth',2); axis([0 10 0 1])
x=[Nf, fliplr(Nf)]; y= [m1.f(1,:) m1.f(end,end:-1:1)]*1000+0.0001;
fill(x,y,ones(1,3)*.9)
axis([0.01 10 1 2*10^3]);
set(gca,'xticklabel','','yticklabel','')
set(gca,'Layer','top','yscale','log','ytick',10.^(-3:4))


