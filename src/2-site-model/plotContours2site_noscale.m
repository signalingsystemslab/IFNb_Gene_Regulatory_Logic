function par = plotContours2site_noscale(c_par, cmap)

    N2 = linspace(0,1,1000);
    I2 = N2;

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
        m1 = calState(m1,N2,I2);
        m1 = calF(m1);
        plotC2(m1,N2,I2);
        colormap(cmap);
        title(tlts{i})
    end
  
end