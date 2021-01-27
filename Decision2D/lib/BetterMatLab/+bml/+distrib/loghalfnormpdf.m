function l = loghalfnormpdf(x, m, s, robustDim)
% p = loghalfnormpdf(x, m, s, robustDim=1)
%
% When robustDim == 1, gives probability mass (sums to 1 given infinite range), 
% rather than pdf.
%
% robustDim is permitted for column vector x only for now.
%    
% 2015 YK wrote the initial version.

l = log(eps) + zeros(size(x));
incl = x >= m;
l(incl) = - log(s .* sqrt(2.*pi)) - (x(incl) - m) .^ 2 ./ (2 .* s .^ 2) ...
          + log(2);
return;

%% Test
x = 2:0.01:10;
m = (rand - 0.5) * 10;
s = rand * 2;

subplot(2,1,1);
plot(x, bml.distrib.loghalfnormpdf(x, m, s), 'b-', ...
     x, log(bml.distrib.halfnormpdf(x(:), m, s, 0)), 'r--');
      
subplot(2,1,2);
plot(x, bml.math.diff_same(bml.distrib.loghalfnormpdf(x, m, s)), 'b-', ...
     x, bml.math.diff_same(log(bml.distrib.halfnormpdf(x(:), m, s, 0))), 'r--');
      
% => diff_same of loghalfnormpdf is very well behaved.
 
%% % ignore robustDim for log.
% 
% if nargin < 4, robustDim = 1; end
% 
%
% if robustDim > 0
%     % Ensures that p sums to 1 given infinite range.
%     assert((robustDim == 1) && iscolumn(x));
%     
%     dx  = mean(diff(x));
%     
%     x1 = [x(1)-dx/2; x(:)+dx/2];
%     x1 = max(x1, m);
%     p  = (normcdf(x1, m, s) - 0.5) * 2;
%     
%     p  = diff(p);
% else
%     p = normpdf(x,m,s) * 2;
% 
%     p(x<m) = 0;
% end
