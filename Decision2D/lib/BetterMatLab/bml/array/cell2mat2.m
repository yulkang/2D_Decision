function m = cell2mat2(c, varargin)
% CELL2MAT2  cell2mat with padding. Return original if non-cell. Enforces each element to be a horizontal vector.
%
% m = cell2mat2(c, ...)
%
% OPTIONS:
% 'pad_with', nan
% 'enforce_double', true
% 'max_len', nan
%
% EXAMPLE:
% >> cell2mat2({[1 2 3], [ 1 3], 1, [], [ 1 2 3 4]})
% ans =
%      1     2     3   NaN
%      1     3   NaN   NaN
%      1   NaN   NaN   NaN
%    NaN   NaN   NaN   NaN
%      1     2     3     4

if ~iscell(c), m = c; return; end

S = varargin2S(varargin, {
    'pad_with', nan
    'enforce_double', true
    'max_len', nan
    });

n = length(c);
l = cellfun(@length, c);

pad_with = S.pad_with;
enforce_double = S.enforce_double;
if isnan(S.max_len)
    max_len = max(l);
else
    max_len = S.max_len;
end

is_nested_cell = cellfun(@iscell, c);
while any(is_nested_cell)
    c(is_nested_cell) = cellfun(@(cc) cc{1}, c(is_nested_cell), ...
        'UniformOutput', false);
    is_nested_cell = cellfun(@iscell, c);
end

if enforce_double
    c2 = cellfun( ...
        @(cc) [double(hVec(cc(1:min(end, max_len)))), ...
               zeros(1, max(0, max_len - length(cc))) + pad_with], ...
        c, ...
        'UniformOutput', false);
else
    c2 = cellfun( ...
        @(cc) [hVec(cc(1:min(end, max_len))), ...
               zeros(1, max_len - length(cc)) + pad_with], ...
        c, ...
        'UniformOutput', false);
end
m  = cell2mat(c2(:));