%savedir = '../Figs/subfigs/'; if ~exist(savedir,'dir'); error('Error:  Wrong save folder.'); end
%datadir = '../Data/';  if ~exist(datadir,'dir'); error('Error: Wrong data folder.'); end
model_3site ='../3-site-model';
model_2site ='../2-site-model';

load(['../data/exp_matrix_norm.mat'])
rsquaredFunc = @(resid) 1 - var(resid,0,2)/var(exp_matrix.ifnb);
color.wt = [.38 .38 .38]; % 262626, blk
color.abc= [255 127 0]/255; % abc , darkorange
color.ac=[255 48 48]/255; %red ,ac
color.ac50 = [139 129 76]/255; % ac50 DEB887
color.irf37 = [2 191 255]/255; % irf37 deepbluesee
color.yellow = [255 193 37]/255; % FFC125
color.gray = [.75 .75 .75];
set(0,'DefaultFigureColormap',flipud(othercolor('RdYlBu_11b',64)))
close all; 
diag_plot = 0; export_plot =0; 
mycolor=[color.gray;
         color.yellow;
         color.yellow/1.5;
         color.yellow/3];
