function res = sev(dat, dim, nanBelow)
% SEV   Standard error of variance.
%
% res = sev(dat, dim, nanBelow)
%
% Based on http://ai.eecs.umich.edu/~fessler/papers/files/tr/stderr.pdf
%
% EXAMPLE:
% % Test sev
% r = randn([1000,1000]); 
% v = var(r);
% sd = std(v);
% disp(sd);
% 
% se = sev(r);
% hist(se);
% disp(median(se));
% hold on;
% plot(sd + [0 0], ylim, 'k-');
%
% % Try replacing r with rand([1000,1000]), i.e., uniform RV,
% % and check that sd does not agree with se any more.
%
% See also VAR, SEM

if ~exist('dim', 'var') || isempty(dim), dim = 1; end
if ~exist('nanBelow', 'var'), nanBelow = 0; end

if numel(dat) ~= length(dat)
    res = var(dat, 0, dim) .* sqrt(2 / (size(dat, dim) - 1));
    
    toNan = size(dat, dim) <= nanBelow;
    res(toNan) = nan;
else
    toNan = length(dat) <= nanBelow;
    if toNan
        res = nan;
    else
        res = var(dat) .* sqrt(2 / (length(dat) - 1));
    end
end
