%% Scatter plot of data points
function ax = plotDataset(data_matrix, cmap, filename)
    figure;
    scatter(data_matrix.irf,data_matrix.nfkb,75,data_matrix.ifnb,"filled", ...
        'MarkerEdgeColor',[0 0 0])
    % cmap = readmatrix("../data/colormap.csv");
    colormap(cmap);
    colorbar;
    ylabel('NFkB');xlabel('IRF');
    ax = gca;
    exportgraphics(ax,filename);
end