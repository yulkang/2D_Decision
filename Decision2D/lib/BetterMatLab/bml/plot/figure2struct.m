function h = figure2struct(src)
% Separates graphics objects in axes into fields of a struct.
%
% h = figure2struct(src)
%
% src : handle of a figure, an axes or handles of its children.
%       gcf by default.
%       When a figure is given, the returning struct h has the same
%       number of rows and columns as the subplots.
%       The rows and columns are determined based on the position,
%       without using subplot. So this function can be used with 
%       figures where the subplots are repositioned, e.g., for printing.
%
% h(row,col) : a struct containing the following fields:
% - axes
% - children (all children regardless of kind)
% - line
% - text
% - legend
% - patch
% - marker (line with LineStyle = 'none'),
% - segment (line with only two coordinates), 
% - segment_vert
% - segment_horz
% - crossline (line that spans at least one axis)
% - nonsegment (lines that connect more than two points).
%
% NOTE:
% To use just one axes out of multiple subplots, use:
%   copyobj(h(row,col).axes, figure);
%   
% To find a subset of the objects, use findobj. For example:
%   red_markers = findobj(h(row,col).marker, 'Color', [1 0 0]);
%
% See also subplot_by_pos, copyobj, findobj

% Yul Kang (c) 2016. hk2699 at columbia dot edu.

if nargin < 1
    src = gcf;
end
if isempty(src)
    h.axes = [];
    h.children = [];
    h.line = [];
    h.text = [];
    h.legend = [];
    h.marker = [];
    h.segment = [];
    h.segment_vert = [];
    h.segment_horz = [];
    h.crossline = [];
    h.nonsegment = [];
    return;
    
elseif strcmpi(get(src(1), 'Type'), 'axes')
    if isscalar(src)
        h.axes = src;
        h.children = get(src, 'Children');
    else
        for ii = numel(src):-1:1
            if isvalidhandle(src(ii))
                h(ii) = bml.plot.figure2struct(src(ii));
            end
        end
        h = reshape(h, size(src));
        return;
    end
    
elseif strcmpi(get(src(1), 'Type'), 'figure')
    assert(isscalar(src), 'Give only one figure at a time!')
    ax = bml.plot.subplot_by_pos(src);
    h = bml.plot.figure2struct(ax);
    return;
else
    h.axes = [];
    h.children = src;
    if ~isempty(h.children)
        src = get(h.children(1), 'Parent');
        h.axes = src;
        assert(strcmpi(get(src, 'Type'), 'axes'), ...
            'Give a figure, axes, or handle graphics objects!');
    end
end
for kind = {'line', 'text', 'legend', 'patch'}
    if ismember(kind{1}, {'legend'})
        if ~isempty(src)
            parent = get(src(1), 'Parent');
        end
    else
        parent = src;
    end

    h.(kind{1}) = findobj(parent, 'Type', kind{1});
end

% Find markers without line
h.marker = findobj(h.line, 'LineStyle', 'none');

% Find line segments (with only two coordinates)
for kind = {'segment', 'segment_vert', 'segment_horz', ...
        'crossline', 'nonsegment'}
    h.(kind{1}) = ghandles(0,0);
end

if ~isempty(src)
    x_lim = xlim(src);
    y_lim = ylim(src);
end

for ii = 1:numel(h.line)
    line1 = h.line(ii);
    x = get(line1, 'XData');
    y = get(line1, 'YData');
    
    if numel(x) == 2
        h.segment(end+1,1) = line1;
        
        if ((x(1) == x_lim(1)) && (x(2) == x_lim(2))) ...
                || ((y(1) == y_lim(1)) && (y(2) == y_lim(2)))
            h.crossline(end+1,1) = line1;
        end
        if x(1) == x(2)
            h.segment_vert(end+1,1) = line1;
        end
        if y(1) == y(2)
            h.segment_horz(end+1,1) = line1;
        end
    elseif ~strcmpi(get(line1, 'LineStyle'), 'none')
        h.nonsegment(end+1,1) = line1;
    end
end
end