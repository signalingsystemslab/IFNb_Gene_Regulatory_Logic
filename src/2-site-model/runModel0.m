% This file is to systematically explore the 8 State model. (5p) 
clear;

%% Set up
% load data
load('../data/exp_matrix_norm.mat') 

% start parallel pool
ncpu=12;
pc=parcluster('local');
pc.NumThreads=2;
parpool(pc,ncpu)
%% Generate synthetic data
npts = 99; % number of synthetic points
nexp = length(exp_matrix.irf); % number of experimental points
new_points = zeros(3,nexp,npts);
data_points = exp_matrix;
rng(2)
%    stdev=sterr is 10% of maximum signal
irf_dev = max(exp_matrix.irf)/10;
irf_var = irf_dev^2;
nfkb_dev = max(exp_matrix.nfkb)/10;
nfkb_var = nfkb_dev^2;
Sigma = [irf_var 0; 0 nfkb_var];

parfor i = 1:npts
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
    new_points(:,:,i) = X;
end

data_points.irf = [data_points.irf,new_points(1,:)];
data_points.nfkb = [data_points.nfkb,new_points(2,:)];
data_points.ifnb = [data_points.ifnb,new_points(3,:)];


ndata = length(data_points.irf); %(npts+1)*7
save('../data/synthetic_data.mat','data_points','new_points','ndata')

%%
% %% model linear eg : 1.enhanceosome 2. oR 3. IRF. 4. NF
% tic
% N= linspace(0,10,1000);
% I = N;
% 
% 
% temp=[ 1 0 0;
%     1 1 1 ;
%     1 1 0;
%     1 0 1 ];
% tlts = {'AND','OR','IRF','NF'};
% figure;
% for i =1:4
%     subplot(2,2,i);
%     pars = temp(i,:);
%     m1 = model0(pars);
%     m1 = calState(m1,N,I);
%     m1 = calF(m1);
%     plotCnorm(m1,N,I);
%     title(tlts{i});
% end
% ax = gcf;
% exportgraphics(ax,'./figs/countour_plot_linear.png');
% toc
% 
% 
% %% check the objfunc 
% pars = [ones(1,1)*.1 ones(1,2)]; 
% % pars =[parsSpace(b,:) 0 0 0 1 0 1];
% [~,rsqred,resid] = objfunc0(pars,exp_matrix,10,1);
% plot(resid,'o-');
% ax = gcf;
% exportgraphics(ax,'residuals.png');
%% let's explore the parameter space. 
ncpars = 1; 
rng(3) % set seed for rand
numbPoints = 10^6; % number of vectors
tvec = npermutek([0 1],2);
ntvec = size(tvec,1);

%
parsSpace = rand(ncpars*numbPoints,1); 
parsSpace= reshape(parsSpace,numbPoints,ncpars);

parsSpace = 10.^((parsSpace-0.5)*8); %1e-4 to 1e4
parsSpace = sort(parsSpace);

% %% minimizing RMSD - slow
% % rmsd =cell(ntvec,1); 
% % resid=cell(ntvec,1);
% 
% rmsd = zeros(numbPoints,1); resid= zeros(numbPoints,7);
% % start parallel
% tic

% 
% for j = 1:ntvec
%    parfor i = 1:numbPoints
%         [rmsd(i),~,resid(i,:)]= objfunc0([parsSpace(i,:) tvec(j,:)],...
%             exp_matrix,10,1);
%         if mod(i,numbPoints/10)==0
%              disp(i)
%          end
%    end
%     disp(j)
%     save(['../data/model0_lin10_normb',num2str(j),'.mat'],...
%         'parsSpace','resid','rmsd')
% end
% 
% % end parallel
% poolobj = gcp('nocreate');
% delete(poolobj);
% toc 

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
save('../data/model0_rmsd_all.mat','minRmsd','minRmsdind','rmsdQuantile', 'minRmsdParam')

tnames = ["t3", "t2", "t1", "t4"];
M = [tnames; minRmsd; minRmsdParam]
save('../data/model0_minimums.mat', 'M');

%% Plot minimum RMSD and corresponding parameter
x=categorical({'AND','NFkB','IRF','OR'});
x=reordercats(x,{'AND','NFkB','IRF','OR'});
subplot(2,1,1)
bar(x,minRmsd)
ylabel('RMSD');

subplot(2,1,2)
bar(x,minRmsdParam)
ylabel('C at min RMSD');
xlabel('model');
sgtitle("Best fitting parameter and RMSD of model0");
ax = gcf;
exportgraphics(ax,'./figs/model0_minRMSD.png');

% %% model best-fitting eg : 1.AND 2. oR 3. IRF. 4. NF
% tic
% N= linspace(0,10,1000);
% I = N;
% % dat = [exp_matrix.irf; exp_matrix.nfkb; exp_matrix.ifnb];
% 
% % min param for each model + model matrix
% % temp=horzcat(minRmsdParam', tvec);
% % tlts = {'AND','NFkB','IRF','OR'};
% % figure;
% % for i =1:4
% %     subplot(2,2,i)
% %     pars = temp(i,:);
% %     m1 = model0(pars);
% %     m1 = calState(m1,N,I);
% %     m1 = calF(m1);
% %     plotCnorm(m1,N,I, exp_matrix);
% %     title(tlts{i})
% % end
% for i = 1:4
%     c_text = sprintf("C= %.3f",minRmsdParam(i));
%     plotContours2site(minRmsdParam(i), cmap);
%     sgtitle(c_text);
%     ax = gcf;
%     fig_name = sprintf("./figs/statespace_c%.3f_model0_minR.png",minRmsdParam(i));
%     exportgraphics(ax,fig_name);
% %     exportgraphics(ax,'./figs/model0_minR_contour.png');
% end
% 
% toc
% 
% figure;
% subplot(2,1,1);
% bar(minRmsd);
% subplot(2,1,2);
% bar(minRmsdParam);
% ax = gcf;
% exportgraphics(ax,'./figs/model0_minRMSD.png');
% 
% % for i =1:4
% %     subplot(2,2,i)
% %     pars = temp(i,:);
% %     m1 = model0(pars);
% %     m1 = calState(m1,N,I);
% %     m1 = calF(m1);
% %     plotCnorm(m1,N,I);
% %     title(tlts{i})
% % end
