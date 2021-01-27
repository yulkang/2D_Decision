function position_subplots(ax, varargin)
% position_subplots(ax, varargin)
%
% OPTIONS
% -------
% 'margin_left',   0.15
% 'margin_bottom', 0.15
% 'margin_right',  0.05
% 'margin_top',    0.1
% ... % margin: a four-vector of [left, bottom, right, top].
% ... % Its non-NaN elements override margin_*
% 'margin', nan(1,4) 
% 'btw_row', 0.05 % if vector, btw_row(1) is between rows 1 and 2.
% 'btw_col', 0.05 % if vector, btw_col(1) is between columns 1 and 2.

% 2016 Yul Kang. hk2699 at columbia dot edu.

S = varargin2S(varargin, {
    'margin_left',   0.15
    'margin_bottom', 0.15
    'margin_right',  0.05
    'margin_top',    0.1
    ... % margin: a four-vector of [left, bottom, right, top].
    ... % Its non-NaN elements override margin_*
    'margin', nan(1,4) 
    'btw_row', 0.05 % if vector, btw_row(1) is between rows 1 and 2.
    'btw_col', 0.05 % if vector, btw_col(1) is between columns 1 and 2.
    ...
    'col_rel', [] % if nonempty, assigns relative width of each column.
    'row_rel', [] % if nonempty, assigns relative height of each row.
    });

if nargin < 1 || isempty(ax)
    ax = bml.plot.subplot_by_pos;
end

n_row = size(ax, 1);
n_col = size(ax, 2);

names = {'left', 'bottom', 'right', 'top'};
for ii = 1:numel(names)
    name = names{ii};
    v = S.(['margin_' name]);
    if isnan(S.margin(ii))
        S.margin(ii) = v;
    end
end

S.btw_row = bml.matrix.rep2fit(S.btw_row(:), [n_row - 1, 1]);
S.btw_col = bml.matrix.rep2fit(S.btw_col(:), [n_col - 1, 1]);

if isempty(S.col_rel)
    S.col_rel = ones(1, n_col);
else
    assert(numel(S.col_rel) == n_col);
end
S.col_rel = S.col_rel ./ sum(S.col_rel);

if isempty(S.row_rel)
    S.row_rel = ones(1, n_row);
else
    assert(numel(S.row_rel) == n_row);
end
S.row_rel = S.row_rel ./ sum(S.row_rel);

width = (1 - sum(S.margin([1, 3])) - sum(S.btw_col)) .* S.col_rel;
height = (1 - sum(S.margin([2, 4])) - sum(S.btw_row)) .* S.row_rel;

for i_row = 1:n_row
    for i_col = 1:n_col
        left = S.margin(1) ...
             + sum(S.btw_col(1:(i_col - 1))) ...
             + sum(width(1:(i_col - 1)));
        bottom = S.margin(2) ...
               + sum(S.btw_row(i_row:(n_row - 1))) ...
               + sum(height((i_row + 1):n_row));
        
        ax1 = ax(i_row, i_col);
        if isa(ax1, 'matlab.graphics.axis.Axes')
            set(ax1, ...
                'Position', [left, bottom, width(i_col), height(i_row)]);
        end
    end
end
end