function C = style_data(varargin)
% Data figure style

% 2021 Yul Kang. hk2699 at caa dot columbia dot edu.

S = varargin2S(varargin);
C = varargin2C(varargin, {
    'Marker', 'o'
    'MarkerEdgeColor', 'w'
    'LineStyle', 'none'
    'LineWidth', 0.25
    });
if isfield(S, 'Color')
    C = varargin2C(C, {
        'MarkerFaceColor', S.Color
        });
end
end