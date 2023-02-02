function par = plotContours2site(c_par, cmap, scale)
%   plots contour plots for the four models along state space 
    tic
    if nargin < 3
        scale = 1; % max IRF=max NFkB=1
    end

    N= linspace(0,1*scale,1000);
    I = N;
    par=[c_par 0 0;
        c_par 1 1 ;
        c_par 1 0;
        c_par 0 1 ];

    tlts = {'AND','OR','IRF','NFkB'};
    figure;
    for i=1:4
        subplot(2,2,i)
        pars = par(i,:);
        m1 = model0(pars);
        m1 = calState(m1,N,I);
        m1 = calF(m1);
        plotC(m1,N,I);
        colormap(cmap);
        title(tlts{i})
    end
    toc
end