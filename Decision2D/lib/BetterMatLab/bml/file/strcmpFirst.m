function res = strcmpFirst(a, b, varargin)
% Compare first parts of two strings.
%
% res = strcmpFirst(a, b, ['opt1', opt1, ...])
%
% a is a string
% b can be either a string or a cell array of strings.
%
% OPTIONS:
% 'mark_shorter_b_different' or 'strict'
% : If false (default), returns if the shorter of a and b matches 
%   the other's beginning.
%   If true, if b is shorter than a, returns false.
%
% EXAMPLE:
% >> strcmpFirst('ab', {'a', 'ab', 'abc'})
% ans =
%      1     1     1
% 
% >> strcmpFirst('ab', {'a', 'ab', 'abc'}, 'mark_shorter_b_different', true)
% ans =
%      0     1     1
%
% See also: strcmpFirsts, strcmpStart, str, PsyLib
%
% 2014 (c) Yul Kang. See help PSYLIB for the license.


S = varargin2S(varargin, { ...
    'strict', []
    'mark_shorter_b_different', false
    }, true);

if ~isempty(S.strict)
    S.mark_shorter_b_different = S.strict;
end

if ischar(b)
    lenA = length(a);
    lenB = length(b);

    if lenA <= lenB
        res = strcmp(a, b(1:lenA));
        
    elseif S.mark_shorter_b_different
        res = false;
        
    else
        warning('Obsolete! Use strcmpStart instead.');
        res = strcmp(a(1:lenB), b);
    end
elseif iscell(b)
    warning('Obsolete! Use strcmpStart instead.');
    res = cellfun(@(bb) strcmpFirst(a, bb, varargin{:}), b);
elseif isempty(b)
    res = false;
else
    error('b must be char, cell, or empty!');
end