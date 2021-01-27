function hgl = gltitle(h, op, t, varargin)
% Column and row titles
%
% hgl = gltitle(h, op, t, varargin)
%
% h     : Array of axes
% op    : 'all', 'row', 'col'
% t     : Title. In case of 'all', a string. In case of 'row' or 'col', a cell array of strings.
%
% OPTIONS:
% 'shift', [0.05, -0.05] % [xshift, yshift] or [xshift, yshift, zshift]. Defaults to [0.05, -0.05].
% 'title_args', {
%     'FontSize',     fntsiz + 8 % or 5
%     'Position',     [0.5, 1.06, 0.5] % [x, y, z]
%     }

 
S = varargin2S(varargin, {
    'shift', [0.05, -0.05] % [xshift, yshift] or [xshift, yshift, zshift]. Defaults to [0.05, -0.05].
    'title_args', {}
    });

if ~isempty(S.shift)
    shiftpos(h, S.shift);
end
if ischar(t), t = {t}; end
        
fntsiz = get(0, 'DefaultTextFontSize');

switch op
    case 'all'
        C = varargin2C(S.title_args, {
            'FontSize',     fntsiz + 8
            'Position',     [0.5, 1.06, 0.5] % [x, y, z]
            });
        
        hgl = glaxes(h, 'title', t, C{:});
        
    case 'row'
        n = size(h, 1);
        if verLessThan('matlab', '8.4')
            hgl = zeros(1, n);
        else
            hgl = gobjects(1, n);
        end
        
        C = varargin2C(S.title_args, {
            'FontSize',             fntsiz + 5
            'HorizontalAlignment',  'right'
            'VerticalAlignment',    'middle'
            'Position',             [-0.15, 0.5, 0]
            'FontWeight',           'bold'
            'Rotation',             0
            });
        
        for ii = 1:n
            hgl(ii) = glaxes(h(ii,:), 'ylabel', t{ii}, C{:});
        end
        
    case 'col'
        n = size(h, 2);
        if verLessThan('matlab', '8.4')
            hgl = zeros(1, n);
        else
            hgl = gobjects(1, n);
        end
        
        C = varargin2C(S.title_args, {
            'FontSize',             fntsiz + 5
            'HorizontalAlignment',  'center'
            'Position',             [0.5 1.01, 0]
            });
        
        for ii = 1:n
            hgl(ii) = glaxes(h(:,ii), 'title', t{ii}, C{:});
        end
end