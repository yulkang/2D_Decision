function[p,varargout] = page(x)
% PAGE       Compute Page test statistic
% INPUTS   : x - n*k data matrix, subjects in rows, treatments in cols
% OUTPUTS  : p - Page test statistic 
%            r - (optional) n*k matrix of x(i,j) midranks
% EXAMPLE  : x = rand(2,3), [p,r] = page(x) 
% SEE ALSO : MCPAGE
% AUTHOR   : Dimitri Shvorob, dimitri.shvorob@vanderbilt.edu, 3/25/07

if nargin < 1
   error('Input argument "x" is undefined')
end
if ~isnumeric(x)
   error('Input argument "x" must be numeric')
end
if ndims(x) ~= 2
   error('Input argument "x" must be a matrix')
end
[n,k] = size(x);
if n == 1
   warning('Only one subject present in "x"')    %#ok
end
if k == 1
   warning('Only one treatment present in "x"')  %#ok
end
r = nan*x;
for i = 1:n
    r(i,:) = tiedrank(x(i,:));
end    
p = sum((1:k).*sum(r,1));
if nargout > 1
   varargout{1} = r;
end   