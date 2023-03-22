function par = plotContourSingle2site(c_par, j, cmap, scale)
%   plots contour plots for the four models along state space 
    tic
    if nargin < 4
        scale = 1; % max IRF=max NFkB=1, assumes not in saturating regime
    end

    N= linspace(0,1*scale,1000);
    I = N;
    tnames = ["AND", "NFkB", "IRF", "OR"];
    tvec = npermutek([0 1],2);

    par=horzcat(repelem(c_par,length(tvec))',tvec);

    t_text = sprintf("Contour plot for best fit %s model", tnames(j));
    disp(t_text);
    c_text = sprintf("C= %.3f",c_par);

    figure;
    pars = par(j,:);
    m1 = model0(pars);
    m1 = calState(m1,N,I);
    m1 = calF(m1);
    plotC(m1,N,I);
    colormap(cmap);
    title(t_text)
    subtitle(c_text)

    toc
end