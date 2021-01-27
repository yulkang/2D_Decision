function pl = terrorbar(x,y,err,varargin)
% pl = terrorbar(x,y,err,varargin)
% error bar without the annoying horizontal line

co = get(gca,'ColorOrder');

if isfield(get(gca),'ColorOrderIndex')
    
    coi = mod ( get(gca,'ColorOrderIndex') , size(co,1) );
    
    pl = plot(x,y,varargin{:});
    
    if not(ismember('color',varargin(1:2:end))) % so that it alternates between colors
        set(pl,'color',co(coi,:));
    end
    
    color = get(pl,'color');
    lw = get(pl,'LineWidth');
    hold on
    for i=1:length(x)
        h = plot([x(i),x(i)],[y(i)-err(i), y(i)+err(i)],'color',color,'LineWidth',lw);
        
        % to ignore in legend:
        leg_info = get(h, 'Annotation');
        y2 = get(leg_info, 'LegendInformation');
        set(y2, 'IconDisplayStyle', 'off');
        
    end
    
    if not(ismember('color',varargin(1:2:end))) % so that it alternates between colors
        set(gca,'ColorOrderIndex',coi+1);
    end
    
else % old matlab
    pl = plot(x,y,varargin{:});
    color = get(pl,'color');
    lw = get(pl,'LineWidth');
    hold on
    for i=1:length(x)
        h = plot([x(i),x(i)],[y(i)-err(i), y(i)+err(i)],'color',color,'LineWidth',lw);
    end
end