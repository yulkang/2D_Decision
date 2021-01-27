function [l, k, theta] = loggampdf_ms(X, m, s, robust_dim)
% [l, k, theta] = loggampdf_ms(X, m, s, robust_dim=0)
%
% m: mean = k * theta
% s: standard deviation = sqrt(k * theta^2)
%
% With shape parameter k an integer, it gives an Earlang distribution,
% the distribution of the sum of k independent variables with
% mean of 1/theta, or rate of theta.
%
% robust_dim: Make p sum to 1 by differentiating cdf.
%             NB: May still need normalization to sum to 1.
%
% EXAMPLE:
% s = 0.0001; t = 0:0.01:1; 
% p1 = gampdf_ms(t, 0.1, s, 2); 
% p2 = gampdf_ms(t, 0.1, s)*0.01; 
% plot(t,p1, 'b.-', t, p2, 'r.-');
%
% 2014 (c) Yul Kang. hk2699 at columbia dot edu.

k = m.^2./(s.^2);
theta = (s.^2)./m;

l = - gammaln(k) - k .* log(theta) + (k - 1) .* log(X) - X ./ theta;

return;

%% Test
x = 0:0.01:5;
m = rand * 10;
s = m * rand;

subplot(2,1,1);
plot(x, bml.distrib.loggampdf_ms(x, m, s), 'b-', ...
     x, log(bml.distrib.gampdf_ms(x(:), m, s, 0)), 'r--');
      
subplot(2,1,2);
plot(x, bml.math.diff_same(bml.distrib.loggampdf_ms(x, m, s)), 'b-', ...
     x, bml.math.diff_same(log(bml.distrib.gampdf_ms(x(:), m, s, 0))), 'r--');

% => diff_same of loggampdf_ms is very well behaved.
 
%%
% if nargin < 4, robust_dim = 0; end
% 
% if robust_dim == 1
%     dX = X(2,:) - X(1,:);
%     X1 = X(end,:) + dX;
%     p = diff(gamcdf(bsxfun(@minus, [X; X1], dX), k, beta), [], 1);
%     
% elseif robust_dim == 2
%     dX = X(:,2) - X(:,1);
%     X1 = X(:,end) + dX;
%     p = diff(gamcdf(bsxfun(@minus, [X, X1], dX), k, beta), [], 2);
%     
% else
%     p = gampdf(X, k, beta);
% end