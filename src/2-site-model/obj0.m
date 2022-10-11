function r=obj0(p)
load('./data/exp_matrix_norm.mat') 
[~,~,r]=objfunc0([10^p 1 0],exp_matrix,10,1);
end