function colores = colors_az(ncolors)
% selects a colormap, and interp linearly

plot_flag = false;

color_scale = 1;

switch color_scale
    case 1
        colors = hsv;
%         close(gcf);
        % colors = circshift(colors,35,1);
        % colors = colors(1:end-12,:);
        remove = [1:6,20:25]; % to avoid too much red and green
        colors = colors(~ismember(1:64,remove),:);
    case 2
        colors = parula;
end

X = linspace(1,ncolors,size(colors,1));
Xq =1:ncolors;
colores = nan(ncolors,3);
for i=1:3
    V = colors(:,i);
    colores(:,i) = interp1(X,V,Xq,'linear');
end


if plot_flag
    for i=1:ncolors
        plot([0,1],[1,1]*i,'color',colores(i,:),'LineWidth',3)
        hold all
    end
end