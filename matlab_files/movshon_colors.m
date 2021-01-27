function colores = movshon_colors(ncolors)
% function colores = movshon_colors(ncolors)
% copied the colors from Goris Neuron 2015, and interpolate linearly
% for more than 5 colors

colors = [144, 42, 143;...
    29, 173, 228;...
    0,164,78;...
    244,127,31;...
    237,32,36] / 255;

if ncolors<=size(colors,1)
    colores = colors(1:ncolors,:);
    
else
    % ncolors = 5;
    X = linspace(1,ncolors,size(colors,1));
    Xq =1:ncolors;
    colores = nan(ncolors,3);
    for i=1:3
        V = colors(:,i);
        colores(:,i) = interp1(X,V,Xq,'linear');
    end
    
    plot_flag= false;
    if plot_flag
        for i=1:ncolors
            plot([0,1],[1,1]*i,'color',colores(i,:),'LineWidth',3)
            hold all
        end
    end
end