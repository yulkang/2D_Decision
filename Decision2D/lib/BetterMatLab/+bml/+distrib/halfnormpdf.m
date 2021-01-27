function p = halfnormpdf(x, m, s, robustDim)
% p = halfnormpdf(x, m, s, robustDim=1)
%
% When robustDim == 1, gives probability mass (sums to 1 given infinite range), 
% rather than pdf.
%
% robustDim is permitted for column vector x only for now.
%    
% 2015 YK wrote the initial version.

if nargin < 4, robustDim = 1; end

if robustDim > 0
    % Ensures that p sums to 1 given infinite range.
    assert((robustDim == 1) && iscolumn(x));
    
    dx  = mean(diff(x));
    
    x1 = [x(1)-dx/2; x(:)+dx/2];
    x1 = max(x1, m);
    p  = (normcdf(x1, m, s) - 0.5) * 2;
    
    p  = diff(p);
else
    p = normpdf(x,m,s) * 2;

    p(x<m) = 0;
end