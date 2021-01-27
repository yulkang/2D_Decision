function [xywh, scr] = set_size(h, siz, corner, varargin)
% [xywh, scr] = set_size(h, [w h], corner, varargin)
% [xywh, scr] = set_size(h, [], corner, varargin)
% [xywh, scr] = set_size(h, {n_row, n_col, row, col}, [], varargin)
%
% corner: 'NW', 'NE', 'SW', 'SE', or ''
% row, col: Can be a vector, like in subplotRC.
%
% OPTIONS:
% 'PaperPositionMode', 'auto'
%
% See also: subplotRC

%% Inputs
if isempty(h), h = gcf; end

scr  = get(0, 'ScreenSize');
xywh = get(h, 'Position');

%% Calculate size
if iscell(siz)
    %% Size for tiling
    n_row = siz{1};
    n_col = siz{2};
    row = siz{3};
    col = siz{4};
    
    siz_cell  = (scr(3:4) - scr(1:2)) ./ [n_col, n_row];
    xywh(3:4) = siz_cell .* [length(col), length(row)];
    xywh(1:2) = siz_cell .* [col(1)-1, n_row - row(1) - length(row) + 1];
  
    
else
    %% Absolute size
    if isempty(siz),
        siz = xywh(3:4); 
    else
        xywh(3:4) = siz; 
    end
    if nargin < 3, corner = ''; end
    
    switch corner
        case ''  
            % Don't move.
        case 'NW'
            xywh(1:2) = [scr(1), scr(4) - siz(2) + 1];

        case 'NE'
            xywh(1:2) = [scr(3) - siz(1) + 1, scr(4) - siz(2) + 1];

        case 'SW'
            xywh(1:2) = scr(1:2);

        case 'SE'
            xywh(1:2) = [scr(3) - siz(1) + 1, scr(2)];

        case 'N'
            xywh(1:2) = [mean(scr([1,3])) - siz(1)/2, scr(4) - siz(2) + 1];

        case 'S'
            xywh(1:2) = [mean(scr([1,3])) - siz(1)/2, scr(2)];        
    end
end

%% Correct
if exist('IsWin', 'file') && IsWin
    xywh(2) = xywh(2) - 50;
end

%% Other properties
S = varargin2S(varargin, {
    'Position',          xywh
    'PaperPositionMode', 'manual'
    'PaperUnits',        'inches'
    });
%     'PaperPositionMode', 'auto'

S.PaperPosition = [0 0 S.Position(3:4) / 72]; % Assuming 72 dpi

%% Handle docked windows - just set the PaperPosition
if strcmpi(get(h, 'WindowStyle'), 'Docked')
    S = rmfield(S, 'Position');
end

%% Actual setting
C = S2C(S);
set(h, C{:});

