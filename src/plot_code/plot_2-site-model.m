%% Make RMSD plot w/ error bars
load("../data/model0_bootstrap_mins.mat")

% NaN means S' * beta = 0, model cannot lead to transcription
% Eliminating resamples for now
nan_locs=isnan(minVals.rmsd(:,1));
minVals.rmsd = minVals.rmsd(~nan_locs,:);
minVals.param = minVals.param(~nan_locs,:);
rmsd = mean(minVals.rmsd);
param = mean(minVals.param);

% Calculate error
rmsd_std = std(minVals.rmsd);
par_std = std(minVals.param);

% rmsd_err = std(minVals.rmsd)/sqrt(length(minVals.rmsd));
% par_err = std(minVals.param)/sqrt(length(minVals.param));

x=categorical({'AND','NFkB','IRF','OR'});
x=reordercats(x,{'AND','NFkB','IRF','OR'})
subplot(2,1,1)
bar(x,rmsd)
hold on
er = errorbar(x,rmsd,rmsd_std);
er.Color = [0 0 0];
er.LineStyle = "none";
ylabel('RMSD');

subplot(2,1,2)
bar(x,param)
hold on
er = errorbar(x,param,par_std);
er.Color = [0 0 0];
er.LineStyle = "none";
ylabel('RMSD');
ylabel('C at min RMSD');
xlabel('model');

%% Explore state space at different parameters
cmap = readmatrix("../data/colormap.csv");
plotContours2site(0.63424, cmap)



