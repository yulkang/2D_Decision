function [s, c] = fieldOrder(s, f1, op, f2)
% [s, c] = fieldOrder(s, f1, op, f2)
%
% s: struct.
% op: 'bef', 'aft', 'first', or 'last'.
% f1, f2: existing field names. f1 can be a cell array of strings.
% c: field names in the final order.
%
% See also: strOrder
%
% 2015 (c) Yul Kang. hk2699 at cumc dot columbia dot edu.

if nargin < 4, f2 = ''; end

c = strOrder(fieldnames(s)', f1, op, f2);
s = orderfields(s, c);
