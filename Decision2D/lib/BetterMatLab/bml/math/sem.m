function res = sem(dat, dim, nanBelow)
% SEM   Standard error of mean.
%
% res = sem(dat, dim, nanBelow)
%
% See also MEAN, STD

if ~exist('dim', 'var') || isempty(dim), dim = 1; end
if ~exist('nanBelow', 'var'), nanBelow = 0; end

if numel(dat) ~= length(dat)
    res = std(dat, 0, dim) / sqrt(size(dat, dim));
    
    toNan = size(dat, dim) <= nanBelow;
    
    res(toNan) = nan;
else
    res = std(dat) / sqrt(length(dat));
end