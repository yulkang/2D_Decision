function [sk, st, m] = skew_distrib(p, v, d)
% [sk, st, m] = skew_distrib(prob, [value=1:size(prob,dim), dim=1])
%
% When value and probability are of different sizes, specify dimension 
% along which to average.
%
% sk: skew (moment coefficient of skewness).
%     See http://en.wikipedia.org/wiki/Skewness#Pearson.27s_moment_coefficient_of_skewness
% st: standard deviation
% m: mean
%
% See also mean_distrib, std_distrib

if nargin < 3 || isempty(d)
    d = 1;
end
if nargin < 2 || isempty(v)
    v = (1:size(p, d))';
end

m3      = mean_distrib(p, v.^3, d);
[st, m] = std_distrib( p, v,    d);

sk = sqrt( m3 - st.^3 );
