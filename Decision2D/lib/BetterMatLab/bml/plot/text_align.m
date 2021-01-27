function h = text_align(txt, varargin)
% h = text_align(txt, varargin)
%
% OPTIONS
% -------
% 'corner', 'NW'
% 'margin', [0.1, 0.1]
% 'margin_unit', 'proportion'
% 'text_props', {}
% 'h_txt', [] % If given, update its position rather than creating new.
%
% 2015 (c) Yul Kang. hk2699 at columbia dot edu.

assert(ischar(txt) || iscell(txt));
S = varargin2S(varargin, {
    'corner', 'NW'
    'margin', [0 0] + 0.025
    'margin_unit', 'proportion'
    'text_props', {}
    'h_txt', [] % If given, update its position rather than creating new.
    });
text_props_default = varargin2S({
    'FontSize', 12
    });

% Determine margin
switch S.margin_unit
    case 'proportion'
        x_lim = xlim;
        y_lim = ylim;
        
        margin = S.margin;
        if isscalar(margin)
            margin = [0 0] + margin; 
        else
            assert(numel(margin) == 2);
        end
        
        margin(1) = margin(1) * diff(x_lim);
        margin(2) = margin(2) * diff(y_lim);
    otherwise
        error('margin_unit=%s not supported!\n', S.margin_unit);
end

% Determine x, y, alignments
assert(ismember(S.corner, {'NW', 'NE', 'SW', 'SE'}));
switch S.corner(1)
    case 'N'
        text_props_default.VerticalAlignment = 'top';
        y = y_lim(2) - margin(2);
    case 'S'
        text_props_default.VerticalAlignment = 'bottom';
        y = y_lim(1) + margin(2);
end
switch S.corner(2)
    case 'W'
        text_props_default.HorizontalAlignment = 'left';
        x = x_lim(1) + margin(1);
    case 'E'
        text_props_default.HorizontalAlignment = 'right';
        x = x_lim(2) - margin(1);
end
C = varargin2C(S.text_props, text_props_default);
if ~isvalidhandle(S.h_txt)
    h = text(x, y, txt, C{:});
else
    h = S.h_txt;
    set(h, 'String', txt, 'Position', [x, y, 0]);
end
end