function lim = beautify_lim(varargin)
% lim = beautify_lim(varargin)
%
% 'xy', 'y'
% 'ax', gca
% 'margin', 0.05

% 2016 Yul Kang. hk2699 at columbia dot edu.

S = varargin2S(varargin, {
    'xy', 'y'
    'axis_dst', []
    'ax', gca
    'margin', 0.1
    ... % 'lim_from'
    ... % - 'data': get limits based on the all coordinates, including error bars'
    ... % - 'markers': get limits based on markers but not error bars
    ... % - 'lim': get limits from the existing lim
    'lim_from', 'data' % 'data'|'markers'|'lim'|numeric vector
    });

if isempty(S.axis_dst), S.axis_dst = S.xy; end
if ~isscalar(S.ax)
    for ii = 1:numel(S.ax)
        C = varargin2C({
            'ax', S.ax(ii)
            }, varargin);
        bml.plot.beautify_lim(C{:});
    end
    return;
end

if isnumeric(S.lim_from)
    assert(isvector(S.lim_from));
    coord = S.lim_from;
elseif ischar(S.lim_from)
    switch S.lim_from
        case 'data'
            xy = bml.plot.get_all_xy(S.ax);
            coord = xy(:, lower(S.xy) == 'xy');
            coord = coord(isfinite(coord));

        case 'markers'
            hs = bml.plot.figure2struct(S.ax);
            hs1 = setdiff(hs.marker, hs.segment);
            coord = cell2vec(get(hs1, [upper(S.xy), 'Data']));

        case 'lim'
            switch S.xy
                case 'x'
                    coord = xlim(S.ax);
                case 'y'
                    coord = ylim(S.ax);
            end
    end
end
if isempty(coord)
    % Do nothing
    return;
end
max_lim = max(coord);
min_lim = min(coord);
range_lim = max_lim - min_lim;
lim = [max_lim - range_lim * (1 + S.margin), ...
       min_lim + range_lim * (1 + S.margin)];

name = [upper(S.axis_dst), 'Lim'];
if lim(1) < lim(2)
    set(S.ax, name, lim);
end
end
