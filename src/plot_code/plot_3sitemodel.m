addpath("../3-site-model/")
%% load exp matrix
load('../data/exp_matrix_norm.mat') 
cmap = readmatrix("../data/colormap.csv");

% Old plotting code, rmoved to R
% %% Scatter plot of synthetic data points
% load("../data/synthetic_data.mat")
% figure;
% scatter(new_points(1,:),new_points(2,:),25,new_points(3,:),"filled")
% hold on
% scatter(exp_matrix.irf,exp_matrix.nfkb,50,exp_matrix.ifnb,"filled", ...
%     'MarkerEdgeColor',[0 0 0], 'LineWidth',2.0)
% colormap(cmap);
% colorbar;
% ylabel('NFkB');xlabel('IRF');
% hold off
% ax = gca;
% exportgraphics(ax,'../2-site-model/figs/syn_data.png');
% 
% % %% Make RMSD plot w/ error bars
% load("../data/model0_bootstrap_mins.mat")
% 
% % NaN means S' * beta = 0, model cannot lead to transcription
% % Eliminating resamples for now
% nan_locs=isnan(minVals.rmsd(:,1));
% minVals.rmsd = minVals.rmsd(~nan_locs,:);
% minVals.param = minVals.param(~nan_locs,:);
% rmsd = mean(minVals.rmsd);
% param = mean(minVals.param);
% 
% % Calculate error
% rmsd_std = std(minVals.rmsd);
% par_std = std(minVals.param);
% 
% % rmsd_err = std(minVals.rmsd)/sqrt(length(minVals.rmsd));
% % par_err = std(minVals.param)/sqrt(length(minVals.param));
% 
% x=categorical({'AND','NFkB','IRF','OR'});
% x=reordercats(x,{'AND','NFkB','IRF','OR'});
% subplot(2,1,1)
% bar(x,rmsd)
% hold on
% er = errorbar(x,rmsd,rmsd_std);
% er.Color = [0 0 0];
% er.LineStyle = "none";
% ylabel('RMSD');
% 
% subplot(2,1,2)
% bar(x,param)
% hold on
% er = errorbar(x,param,par_std);
% er.Color = [0 0 0];
% er.LineStyle = "none";
% ylabel('RMSD');
% ylabel('C at min RMSD');
% xlabel('model');

%% Compare best parameters for experimental data
load('../data/bestFit_3site_b1.mat')
%  Sort pars by best RSS
[rss_sorted,rss_order]=sort(rss);
parsFinal_sorted = parsFinal(rss_order,:);
pars_normalized = parsFinal_sorted./max(parsFinal_sorted);
xvalues = {'t_1','t_2','t_3','t_4','t_5','t_6'};

R=linspace(1,0,50);
G=linspace(1,0,50);
B=linspace(1,0.7,50);
bluemap=colormap([R(:), G(:), B(:)]);

imagesc(pars_normalized);
colormap(bluemap);
colorbar;
set(gca, 'XTickLabel',xvalues);
ylabel("Starting paramater set")
xlabel("Param")
title('Best parameter values sorted by RSS');

ax = gcf;
fig_name = sprintf("../3-site-model/figs/exp_data_bestParams.jpg");
exportgraphics(ax,fig_name);
%% Compare best parameters for synthetic data (heatmap)
load('../data/pars_3site_b1_syndata_small.mat')
%  Sort pars by best RSS
[rmsd_sorted,rmsd_order]=sort(min_rmsd_syn);
pars_sorted = parsFinal(rmsd_order,:);
pars_normalized = pars_sorted./max(pars_sorted);
xvalues = {'t_1','t_2','t_3','t_4','t_5','t_6'};

R=linspace(1,0,50);
G=linspace(1,0,50);
B=linspace(1,0.7,50);
bluemap=colormap([R(:), G(:), B(:)]);

imagesc(pars_normalized);
colormap(bluemap);
colorbar;
set(gca, 'XTickLabel',xvalues);
ylabel("Synthetic data set")
xlabel("Param")
title('Best parameter values sorted by RMSD');

ax = gcf;
fig_name = sprintf("../3-site-model/figs/syn_data_bestParams.jpg");
exportgraphics(ax,fig_name);

%% Load best values of t and plot contour plot
N= linspace(0,1*scale,1000);
I = N;
p=parsFinal_sorted(1,:);
m = modelB1(p);
% transfer the experimental NFkB and IRF data 
m = calState(m,N,I);
m = calF(m);

plotCnorm(m, N, I);
colormap(cmap);
title('Best fit for Model \beta 1','FontSize',20);

ax = gcf;
fig_name = sprintf("../3-site-model/figs/exp_data_best_contour.png");
exportgraphics(ax,fig_name);

%% Explore best and worst fits
load("../data/interesting_values.mat")
load('../data/synthetic_data.mat')

for i=1:length(labs)
    k=vals(i,1);
    c=vals(i,2);
    fprintf("%s, Dataset#%.0f, RMSD=%.3f, C=%.4f\n", labs(i), k, c, ...
        vals(i,3))
    % Plot dataset
    data_matrix.irf = new_points(1,:,k);
    data_matrix.nfkb = new_points(2,:,k);
    data_matrix.ifnb = new_points(3,:,k);
    plotDataset(data_matrix,cmap,append("../2-site-model/figs/",labs(i),"_dataset.png"));

    % Plot state space
    plotContours2site(c, cmap)
    ax = gcf;
    fig_name = append("../2-site-model/figs/",labs(i),"_statespace.png");
    exportgraphics(ax,fig_name);
end