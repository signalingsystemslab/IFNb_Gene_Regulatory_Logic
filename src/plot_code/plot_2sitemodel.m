addpath("../2-site-model/")
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

%% Explore state space at different parameters
cmap = readmatrix("../data/colormap.csv");
c_pars = [0.001, 0.005, 0.01, 0.1, 0.5, 1, 1.5, 2, 5, 10];
% c_pars = [10^-4];

% fprintf("Starting\n")
% ncpu=4;
% pc=parcluster('local');
% pc.NumThreads=2;
% parpool(pc,ncpu);

for i = 1:length(c_pars)
    tic
    fprintf("Printing contour plots for c=%.5f\n", c_pars(i))
    plotContours2site(c_pars(i), cmap);
    ax = gcf;
    fig_name = sprintf("../2-site-model/figs/statespace_c%.3f.png",c_pars(i));
    exportgraphics(ax,fig_name);
    fprintf("Done with one (more)\n")
end

%% Load best values of C and plot contour plot
load('../data/model0_rmsd_all.mat')
% I N
% 0 0 AND
% 0 1 NFkB
% 1 0 IRF
% 1 1 OR

for j=1:length(tvec)
    c_par=minRmsdParam(j);
    plotContourSingle2site(c_par, j, cmap)
    ax = gcf;
    fig_name = sprintf("../2-site-model/figs/statespace_%s_fit.png",tnames(j));
    exportgraphics(ax,fig_name);
end

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