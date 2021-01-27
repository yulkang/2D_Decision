function [e, y] = sev_distrib(p, v, d)
% [e, y] = sev_distrib(prob, [value=1:size(prob,dim), dim=1])
%
% When value and probability are of different sizes, specify dimension 
% along which to average.
%
% p should be a histogram, i.e., sum(p, d) should be the number of samples.
%
% s: standard error of variance
% m: variance
%
% See also mean_distrib, sem_distrib, skew_distrib

if nargin < 3 || isempty(d)
    d = 1;
end
if nargin < 2 || isempty(v)
    v = (1:size(p,d))';
end

n = sum(p, d);
y = bml.stat.var_distrib(p, v, d);
e = y .* sqrt(2 ./ (n - 1));
