function m = mean_distrib(p, v, d)
% m = mean_distrib(prob, [value=1:size(prob,dim), dim=1])
%
% When value and probability are of different sizes, specify dimension 
% along which to average.
%
% See also std_distrib, sem_distrib

if nargin < 3 || isempty(d)
    d = 1;
end
if nargin < 2 || isempty(v)
    v = reshape2vec(1:size(p, d), d);
end

if nargin >= 3
    m = sum(bsxfun(@times, v, bsxfun(@rdivide, p, sum(p, d))), d);
else
    try
        m = sum(v .* (p ./ sum(p)));
    catch
        m = sum(bsxfun(@times, v, bsxfun(@rdivide, p, sum(p))));
    end
end