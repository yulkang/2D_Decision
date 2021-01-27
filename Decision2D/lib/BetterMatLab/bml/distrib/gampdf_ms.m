function [p, k, beta] = gampdf_ms(X, m, s, robust_dim)
% [p, k, theta] = gampdf_ms(X, m, s, robust_dim=0)
%
% m: mean = k * theta
% s: standard deviation = sqrt(k * theta^2)
%
% With shape parameter k an integer, it gives an Earlang distribution,
% the distribution of the sum of k independent variables with
% mean of 1/theta, or rate of theta.
%
% robust_dim: Make p sum to 1 by differentiating cdf and dividing by sum.
%
% EXAMPLE:
% s = 0.0001; t = 0:0.01:1; 
% p1 = gampdf_ms(t, 0.1, s, 2); 
% p2 = gampdf_ms(t, 0.1, s)*0.01; 
% plot(t,p1, 'b.-', t, p2, 'r.-');
%
% 2014 (c) Yul Kang. hk2699 at columbia dot edu.

if nargin < 4, robust_dim = 0; end

k = m.^2./(s.^2);
beta = (s.^2)./m;

if robust_dim == 1
    dX = X(2,:) - X(1,:);
    X1 = X(end,:) + dX;
    p = diff(gamcdf(bsxfun(@minus, [X; X1], dX), k, beta), [], 1);
    p = bsxfun(@rdivide, p, sum(p));
    
elseif robust_dim == 2
    dX = X(:,2) - X(:,1);
    X1 = X(:,end) + dX;
    p = diff(gamcdf(bsxfun(@minus, [X, X1], dX), k, beta), [], 2);
    p = bsxfun(@rdivide, p, sum(p, 2));
    
else
    assert(robust_dim == 0, ...
        'robust_dim=%d not supported yet!', robust_dim);
    
    p = gampdf(X, k, beta);
end