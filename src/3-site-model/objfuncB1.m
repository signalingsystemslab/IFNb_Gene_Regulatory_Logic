function [rmsd,rsquared,residues]= objfunc0(pars,exp_matrix,scale,nflag)

    if nargin<3
        scale =10;
    elseif isempty(scale)
        scale =10; 
    elseif nargin <4
        nflag = 0; % normalize
    end
    
    m = model0(pars);
    % transfer the experimental NFkB and IRF data 
    m = calState2(m,exp_matrix.nfkb*scale,exp_matrix.irf*scale);
    m = calF2(m); 
    if nflag
        residues = m.f/max(m.f(:)) - exp_matrix.ifnb';
    else
        residues = m.f - exp_matrix.ifnb;
    end
    
    rmsd = sqrt(sum(residues.^2)/numel(residues));
    rsquared = 1 - var(residues)/var(exp_matrix.ifnb); 
end