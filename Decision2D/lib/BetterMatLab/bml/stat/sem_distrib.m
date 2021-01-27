function [s, m] = sem_distrib(p, v, d)
% [s, m] = sem_distrib(count, [value=1:size(prob,dim), dim=1])
%
% When value and count are of different sizes, specify dimension 
% along which to average.
%
% s: standard error of mean
% m: mean
%
% See also mean_distrib, std_distrib

if nargin < 2, v = []; end
if nargin < 3, d = 1; end

[s, m] = bml.stat.std_distrib(p, v, d);

s = bsxfun(@rdivide, s, sqrt(sum(p, d)));

