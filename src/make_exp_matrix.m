exp_matrix.irf = [0.1, 0.25, 0.01, 0.75, 1, 0.075, 0];
exp_matrix.nfkb = [1,   0,    1,    0.5, 0, 0.5, 0.5];
exp_matrix.ifnb = [0.4, 0.2,  0,    1,   1, 0.4, 0];

exp_matrix_old.irf = [0.1000 0.2000 0.0100 0.6000 0.8000 0.1000 0];
exp_matrix_old.nfkb = [0.5000 0 1 0.5000 0 1 1];
exp_matrix_old.ifnb = [0.4000 0.2000 0 1 1 0.4000 0];

save('./data/exp_matrix_norm.mat', 'exp_matrix')

% Convert to matrix with columns for irf, nfkb, ifnb
exp_matrix_mat = [exp_matrix.irf', exp_matrix.nfkb', exp_matrix.ifnb'];
% Save as csv
csvwrite('./data/exp_matrix_norm.csv', exp_matrix_mat)