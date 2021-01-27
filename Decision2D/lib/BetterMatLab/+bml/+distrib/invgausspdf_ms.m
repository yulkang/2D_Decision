function p = invgausspdf_ms(x, mu, sd, robust_dim)
% p = invgausspdf_ms(x, mu, sd, robust_dim)
persistent pd

if nargin < 4
    robust_dim = 0;
end

if isempty(pd)
    pd = makedist('InverseGaussian');
end
pd.mu = mu;
pd.lambda = mu.^3 ./ sd.^2;

if robust_dim == 0
    p = pd.pdf(x);
elseif robust_dim == 1
%     dx = x(2) - x(1);
%     x = [x(1) - dx / 2; x(:) + dx / 2];
%     
%     p = diff(pd.cdf(x));
    
%     if any(isnan(p))
        p = pd.pdf(x);
        p = bsxfun(@rdivide, p, sum(p));
%     end 
else
    error('Unsupported robust_dim=%s\n', robust_dim);
end
end