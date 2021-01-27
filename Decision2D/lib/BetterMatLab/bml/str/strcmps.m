function [ix intersectStrs] = strcmps(template, querry, ignoreCase)
% STRCMPS   Returns whether each QUERRY is in TEMPLATE.
%
% [ix, intersectStrs] = strcmps(template, querry, [ignoreCase=false])
%
% TEMPLATE is always a cell array of strings.
% QUERRY is a string or a cell array of strings.
%
% ix is a logical vector of the same length as QUERRY.
% intersectStrs is QUERRY(ix).
%
% EXMAPLE:
% >> strcmps({'a', 'b', 'c'}, {'a', 'b', 'd', 'e'})
% ans =
%      1     1     0     0
%
% See also STRCMPFINDS, str, PsyLib
%
% 2013 (c) Yul Kang. See help PsyLib for the license.

if nargin >=3 && ignoreCase,
    if ischar(template)
        ix = strcmpi(template, querry);
    else
        ix = false(size(querry));
        for ii = 1:length(template)
            ix = ix | ix(strcmpi(template{ii}, querry));
        end
    end
else
    if ischar(template)
        ix = strcmp(template, querry);
    else
        ix = false(size(length(querry)));    
        for ii = 1:length(template)
            ix = ix | strcmp(template{ii}, querry);
        end
    end
end

if nargin >=2
    intersectStrs = querry(ix);
end