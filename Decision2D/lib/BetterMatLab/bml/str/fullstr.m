function s = fullstr(sep, varargin)
% FULLSTR : Connects strings with sep. Unlike str_bridge, puts '_' even when empty.
%
% EXAMPLE:
% >> fullstr('_', 'a', 'b', '', 'c')
% ans =
% a_b__c
%
% See also: strsep, funPrintf, str_bridge

s = sprintf(['%s', sep], varargin{:});
s = s(1:(length(s)-length(sep)));