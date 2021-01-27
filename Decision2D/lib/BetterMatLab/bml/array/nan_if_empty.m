function v = nan_if_empty(v)
% NAN_IF_EMPTY - Returns nan if empty.
%
% mat2  = nan_if_empty(mat1)
% cell2 = nan_if_empty(cell1)
%
% For a matrix input, returns NaN if the input is empty.
% For a cell array input, fills NaN in the cells that has [].
%
% 2013 (c) Yul Kang, hk2699 at columbia dot edu.

if iscell(v)
    v = cellfun(@(c) nan_if_empty(c), v, 'UniformOutput', false);
elseif isempty(v)
    v = nan;
end