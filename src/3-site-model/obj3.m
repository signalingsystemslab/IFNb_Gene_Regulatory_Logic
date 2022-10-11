function r=obj3(p)
load('../exp_matrix_norm.mat') 
p(1:2) = 10.^p(1:2);
[~,~,r]=objfunc3(p,exp_matrix,10,1);
end