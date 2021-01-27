function y = linspaceN(d1, d2, n, dimRep)
%LINSPACE Linearly spaced vector.
%   LINSPACEN(X1, X2) generates a row vector of 100 linearly
%   equally spaced points between X1 and X2.
%
%   LINSPACEN(X1, X2, N) generates N points between X1 and X2.
%   For N = 1, LINSPACE returns X2.
%
%   LINSPACEN(X1, X2, N, DIM) generates N points between X1 and X2,
%   along DIM.
%
%   Class support for inputs X1,X2:
%      float: double, single
%
%   See also LOGSPACE, COLON.
%
%   Copyright 1984-2011 The MathWorks, Inc.
%   $Revision: 5.12.4.6 $  $Date: 2011/05/17 02:22:58 $
%
%	modified by Yul Kang 2011. hk2699 at caa dot columbia dot edu.

nDim = length(size(d1));

if (nDim~=length(size(d2))) || any(size(d1)~=size(d2))
	error('X1 and X2 must be of the same size!');
end

if nargin == 2
    n = 100;
end

if nargin == 4
	rep = ones(1,max(nDim, dimRep));
	rep(dimRep) = n-2;
else
	if (nDim==2) && all(size(d1)==[1 1])
		dimRep = 2;
		rep = [ones(1,nDim)	n-2];
	elseif (nDim==2) && size(d1,1)==1
		dimRep = 1;
		rep = [n-2 1];
	elseif (nDim==2) && size(d1,2)==1
		dimRep = 2;
		rep = [1 n-2];
	else
		dimRep = nDim+1;
		rep = [ones(1,nDim)	n-2];
	end
end

y = cat(dimRep, d1, ...
				cat(dimRep, ...
					repmat(d1,rep) ...
					+ cumsum( ...
								repmat((d2-d1)/double(n-1), rep), ...
								dimRep ...
							), ...
					d2 ...
					) ...
		);
