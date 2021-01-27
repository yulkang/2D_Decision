function c = vecOnDim(cDim, nDim)
% VECONDIM  Gives a cell vector to feed reshape to make a vector on a dimension.
%
% c = vecOnDim(cDim, [nDim = max(cDim,2)]);
%
% Example 1:
%   vecOnDim(1)
% ans =
%   []  [1]
%
% Example 2:
%   vecOnDim(3)
% ans = 
%   [1] [1] []
%
% Example 3:
%   vecOnDim(2, 4)
% ans =
%   [1] [] [1] [1]
%
% Example 4:
%   reshape(zeros(2,2), ans{:}) % Uses Example 3's output
%                               % Makes a vector on the 2nd dim (=row vector).
% ans =
%   0 0 0 0
%
% See also: reshape2vec

if nargin < 2, nDim = max(cDim, 2); end

c = dimVec(cDim, [], nDim, 1, true);    
end