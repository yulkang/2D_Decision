function h = borde_en_axis(varargin)
% draw a square around the current axis
% borde_en_axis('ax',gca,'color','w','width',0.5)

color = [1 0 0]; width = 0.5; hax = gca;
set(gcf,'currentAxes',hax)
xli = get(gca,'xlim');
yli = get(gca,'ylim');
for i=1:length(varargin)
    if ischar(varargin{i}) && strcmp(varargin{i},'color') 
        color = varargin{i+1};
    elseif ischar(varargin{i}) && strcmp(varargin{i},'width') 
        width = varargin{i+1};
    elseif ischar(varargin{i}) && strcmp(varargin{i},'ax') 
        hax  = varargin{i+1};
        set(gcf,'currentAxes',hax);
    elseif ischar(varargin{i}) && strcmp(varargin{i},'xstart') 
        xli(1) = varargin{i+1};
    elseif ischar(varargin{i}) && strcmp(varargin{i},'ystart') 
        yli(1) = varargin{i+1};
    elseif ischar(varargin{i}) && strcmp(varargin{i},'xend') 
        xli(2) = varargin{i+1};
    elseif ischar(varargin{i}) && strcmp(varargin{i},'yend') 
        yli(2) = varargin{i+1};
    end
end

hold on
h = [];
for i=1:2
    h(end+1) = plot([xli(i) xli(i)],yli,'color',color,'LineWidth',width);
    h(end+1) = plot(xli,[yli(i) yli(i)],'color',color,'LineWidth',width);
end