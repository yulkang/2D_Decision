function [isSpec, propCell] = isLineSpec(str)
% ISLINESPEC    Sees if a string is linespec, and converts it to properties.
%
% [tf, propCell] = isLineSpec(str)
%
% set(gca, propCell{:}) will do similar things as plot(..., str).
% Useful for functions that doesn't support the latter syntax, like cdfplot.
%
% Example:
% [tf, propCell] = isLineSpec('r-')
%
% tf        = 1
% propCell  = {'Color', 'r', 'LineStyle', '-'}
%
% See also: CDFPLOTSPEC

% TODO: parse Marker

lowerPart = (str>='a') & (str<='z');

isSpec = any(strcmps(num2cell(str), {'+', '-', '*', '.'})) ...
        || (nnz(lowerPart) <= 2);

propCell = {};

if isSpec
    if any(lowerPart)
        propCell = [propCell {'Color', str(lowerPart)}];
    end
    
    if any(~lowerPart)
        propCell = [propCell {'LineStyle', str(~lowerPart)}];
    end
end