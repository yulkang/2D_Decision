function xy = get_all_xy(ax, obj_desc)
% xy = get_all_xy(ax, [obj_desc])
%
% xy(:,1) : all x
% xy(:,2) : all y
%
% Get all children's xy
if ~exist('ax', 'var')
    ax = gca;
end
if ~exist('obj_desc', 'var')
    obj_desc = {};
end

if ~isscalar(ax) 
    xy = cell2mat(arrayfun(@(a) bml.plot.get_all_xy(a, obj_desc), ax(:), ...
        'UniformOutput', false));
    return;
end

obj_desc = varargin2C(obj_desc, {
    'Parent', ax
    });

ch = findobj(ax, obj_desc{:});

try
    xs = get(ch, 'XData');
    ys = get(ch, 'YData');
    xy = [vVec([xs{:}]), vVec([ys{:}])];
catch
    n_ch = numel(ch);
    xs = cell(n_ch, 1);
    ys = cell(n_ch, 1);

    for ii = 1:numel(ch)
        xs{ii} = vVec(get(ch(ii), 'XData'));
        ys{ii} = vVec(get(ch(ii), 'YData'));
    end
    xy = [cell2mat(xs), cell2mat(ys)];
end
end