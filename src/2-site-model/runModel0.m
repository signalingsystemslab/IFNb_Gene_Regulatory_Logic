% This file is to systematically explore the 8 State model. (5p) 
clear;

%% Set up
% load data
load('../data/exp_matrix_norm.mat') 

% start parallel pool
ncpu=12;
pc=parcluster('local');
pc.NumThreads=2;
parpool(pc,ncpu);
%% Generate synthetic data
npts = 99; % number of synthetic points
nexp = length(exp_matrix.irf); % number of points in an experiment
new_points = zeros(3,nexp,npts);
data_points = exp_matrix;
rng(2)
%    stdev=sterr is 10% of maximum signal
irf_dev = max(exp_matrix.irf)/10;
irf_var = irf_dev^2;
nfkb_dev = max(exp_matrix.nfkb)/10;
nfkb_var = nfkb_dev^2;
Sigma = [irf_var 0; 0 nfkb_var];

parfor k = 1:npts
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
    new_points(:,:,k) = X;
end

data_points.irf = [data_points.irf,new_points(1,:)]; % not needed
data_points.nfkb = [data_points.nfkb,new_points(2,:)];
data_points.ifnb = [data_points.ifnb,new_points(3,:)];

%  export data to csv
krep = [repelem(0,nexp)];
for k=1:npts
    krep = [krep, repelem(k, nexp)];
end
M = [data_points.irf', data_points.nfkb', data_points.ifnb', krep'];
M = ["IRF", "NFkB", "IFNb", "exp";M];
writematrix(M,'../data/syn_data.csv')


ndata = length(data_points.irf); %(npts+1)*7
save('../data/synthetic_data.mat','data_points','new_points','ndata')
fprintf("Done generating synthetic data \n")

%% Build parameters set 
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
save('../data/pars_model2site.mat','parsSpace')

%% Minimize RMSD
rmsd =cell(ntvec,1); 
resid=cell(ntvec,1);

rmsd = zeros(numbPoints,1); resid= zeros(numbPoints,7);

tic
% save all RMSD values for real data
fprintf("Finding RMSD for exp data \n")
for j = 1:ntvec
   parfor i = 1:numbPoints
        [rmsd(i),~,resid(i,:)]= objfunc0([parsSpace(i,:) tvec(j,:)],...
            exp_matrix,10,1);
        if mod(i,numbPoints/10)==0
             disp(i)
         end
   end
    fprintf("Finished %d models\n", j)
    save(['../data/model0_lin10_normb',num2str(j),'.mat'],...
        'parsSpace','resid','rmsd')
end

% save only minimum RMSD for syn data
fprintf("Finding RMSD for syn data \n")
syn_mins = zeros(ntvec,2,npts); %min rmsd, best C param

for k = 1:npts
    data_matrix.irf = new_points(1,:,k);
    data_matrix.nfkb = new_points(2,:,k);
    data_matrix.ifnb = new_points(3,:,k);

    for j = 1:ntvec
        t = tvec(j,:);
        rmsd_tmp=zeros(numbPoints,1);
        parfor i = 1:numbPoints
            [rmsd_tmp(i),~,]= objfunc0([parsSpace(i,:) t],...
            data_matrix,10,1);
            if mod(i,numbPoints/10)==0
             disp(i)
            end
        end
        [m,i] = min(rmsd_tmp);
        syn_mins(j,1,k) = m;
        syn_mins(j,2,k) = parsSpace(i,:);
    end
    fprintf("%d points\n",k)
end
save("../data/minRMSD_synthetic.mat",'syn_mins')

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
    load(['../data/model0_lin10_normb',num2str(jth),'.mat']);
    [minRmsd(jth), minRmsdind(jth)] =   min(rmsd);
    rmsdQuantile{jth} = rmsd(rmsd<prctile(rmsd,10));
    param = parsSpace(minRmsdind(jth),:);
    minRmsdParam(:,jth) = param;
end
%
save('../data/model0_rmsd_all.mat','minRmsd','minRmsdind','rmsdQuantile', 'minRmsdParam')

%  Export to load in R
t=[3,2,1,4];
M = ["minRMSD", "bestC","model";minRmsd', minRmsdParam',t'];
writematrix(M,'../data/exp_data_mins.csv')

% tnames = ["t3", "t2", "t1", "t4"];
% M = [tnames; minRmsd; minRmsdParam];
% save('../data/model0_minimums.mat', 'M');


trep = [];
for j=1:ntvec
    trep = [trep, repelem(t(j), npts)];
end
M = reshape(syn_mins,[npts*ntvec,2]);
M = horzcat(M, trep');
M = ["minRMSD", "bestC","model"; M];
writematrix(M,'../data/syn_data_mins.csv')

fprintf("Done /n")
% %% Plot minimum RMSD and corresponding parameter
% x=categorical({'AND','NFkB','IRF','OR'});
% x=reordercats(x,{'AND','NFkB','IRF','OR'});
% subplot(2,1,1)
% bar(x,minRmsd)
% ylabel('RMSD');
% 
% subplot(2,1,2)
% bar(x,minRmsdParam)
% ylabel('C at min RMSD');
% xlabel('model');
% sgtitle("Best fitting parameter and RMSD of model0");
% ax = gcf;
% exportgraphics(ax,'./figs/model0_minRMSD.png');

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
