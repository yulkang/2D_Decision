function l = lognormpdf(x, mu, sigma)
% l = lognormpdf(x, mu, sigma)

if nargin < 2
    mu = 0;
end
if nargin < 3
    sigma = 1;
end

l = -1./2 .* log(2 .* pi .* sigma .^ 2) ...
    - (x - mu) .^ 2 ./ (2 .* sigma .^ 2);