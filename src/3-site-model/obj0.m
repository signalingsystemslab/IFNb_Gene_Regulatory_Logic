%  Not used anymore
function r=obj0(p)
% load('../exp_matrix_norm.mat') 
p(1) = 10^p(1);
[~,~,r]=objfunc0(p,exp_matrix,1,1);
end