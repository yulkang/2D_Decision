function hNew = color_points(x, y, varargin)
% hNew = color_points(x, y, varargin)
% hNew = color_points(hOld, [], varargin)
%
% OPTIONS
% -------
% 'colors',    @hsv2
% 'deleteOld', false
% 'plotOpt',   {}
% 'reverseColor', false

S = varargin2S(varargin, {
    'colors',    @hsv2
    'deleteOld', false
    'plotOpt',   {}
    'reverseColor', false
    });

if nargin < 2 || isempty(y)
    hOld = x;

    if isscalar(hOld)
        plotOpt = copyFields(struct, get(hOld), {
            'Marker'
            'MarkerSize'
            'MarkerEdgeColor'
            'LineStyle'
            'LineWidth'
            });
        plotOpt = varargin2C(S.plotOpt, plotOpt);

        x      = hVec(get(hOld, 'XData'));
        y      = hVec(get(hOld, 'YData'));
    else
        hNew   = hOld;
        nColor = numel(hNew);
        colors = S.colors(nColor);
        if S.reverseColor, colors = flipud(colors); end
        for iColor = 1:nColor
            set(hNew(iColor), ...
                'Color', colors(iColor,:), ...
                'MarkerFaceColor', colors(iColor,:));
        end
        return;
    end
%     hAx = get(hOld(1), 'Parent');
else
    hOld = [];
%     hAx = gca;
end

nColor = max(numel(x), numel(y));

x      = rep2fit(x, [1, nColor]);
y      = rep2fit(y, [1, nColor]);

hNew   = ghandles(1, nColor);

if isa(S.colors, 'function_handle')
    colors = S.colors(nColor);
else
    colors = S.colors;
end
if S.reverseColor
    colors = flipud(colors);
end

for iColor = nColor:-1:1
    plotOpt = varargin2C({
        'Color',           colors(iColor,:)
        'MarkerFaceColor', colors(iColor,:)
        }, plotOpt);
    
    hNew(iColor) = plot(x(iColor), y(iColor), plotOpt{:});
    hold on;
end
hold off;

if S.deleteOld && ~isempty(hOld)
    try
        delete(hOld);
    catch err
        warning(err_msg(err));
    end
end
end