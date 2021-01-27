function [ix, intersectStrs] = strcmpfinds(querry, template, ignoreCase, order)
% [ix, intersectStrs] = strcmpfinds(querry, template, [ignorCase=false], order = 'first')
%
% querry   : A string or a cell array of strings.
% template : Always a cell array of strings.
%
% ix       : A numeric vector of the same length as querry, such that
%            querry{ii} == template{ix(ii)}.
%            When no template matches querry{ii}, ix(ii) = nan.
%            When multiple entries match querry{ii}, the first or last index
%            is returned, depending on ORDER.
%
% intersectStrs : template(ix).
%
% order    : Either 'first' or 'last'.
%
% EXAMPLE:
% >> strcmpfinds({'a', 'b', 'c'}, {'a', 'c'})
% ans =
%      1   NaN     2
%
% >> strcmpfinds({'a', 'b', 'c'}, {'a', 'b', 'b'})
% ans =
%      1     2   NaN

if nargin < 4, order = 'first'; end

if ischar(querry)
    if nargin >=3 && ignoreCase
        ix = find(strcmpi(querry, template));
    else
        ix = find(strcmp(querry, template));
    end
else
    if nargin >= 3 && ignoreCase
        ix = cellfun(@(s) nan_if_empty(find(strcmpi(s, template), 1, order)), querry);
    else
        ix = cellfun(@(s) nan_if_empty(find(strcmp(s, template), 1, order)), querry);
    end
end

if nargout >= 2
    intersectStrs = template(ix(~isnan(ix)));
end