function [C, r] = linespec2C(s, varargin)
% Parses string linespec into name-value pairs.
%
% [C, r] = linespec2C(s, ...)
%
% C : name_value pairs.
% r : unparsed remainder.
%
% See also varargin2plot
%
% 2015 (c) Yul Kang. hk2699 at cumc dot columbia dot edu.

if iscell(s)
    if ischar(s{1})
        C = s(2:end);
        s = s{1};
        [l,m,c,r] = parse_linespec(s);
    else
        C = s;
        l = '';
        m = '';
        c = '';
        r = '';
    end
elseif isstruct(s)
    C = S2C(s);
    l = '';
    m = '';
    c = '';
    r = '';    
else
    assert(ischar(s));
    [l,m,c,r] = parse_linespec(s);
    C = {};
end

if ~isempty(l)
    C = [C, {'LineStyle', l}];
end
if ~isempty(m)
    C = [C, {'Marker', m}];
end
if ~isempty(c)
    C = [C, {'Color', c}];
end

C = varargin2C(varargin, C);