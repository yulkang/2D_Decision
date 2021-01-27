function [s, m] = std_distrib(p, v, d)
% [s, m] = std_distrib(prob, [value=1:size(prob,dim), dim=1])
%
% When value and probability are of different sizes, specify dimension 
% along which to average.
%
% s: standard deviation
% m: mean
%
% See also mean_distrib, sem_distrib, skew_distrib

if nargin < 3 || isempty(d)
    d = 1;
end
if nargin < 2 || isempty(v)
    v = (1:size(p, d))';
end

if nargin >= 3
    m  = mean_distrib(p, v,    d);
    m2 = mean_distrib(p, v.^2, d);
else
    m  = mean_distrib(p, v   );
    m2 = mean_distrib(p, v.^2);
end

s  = sqrt( m2 - m.^2 );
