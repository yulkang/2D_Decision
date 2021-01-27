function vec = dimVec(cDim, v, nDim, vOthers, cellOut)
% DIMVEC    Make a vector useful for REPMAT or RESHAPE.
%
% vec = dimVec(cDim, [v = 1, nDim = max(cDim,2), vOthers = 0])
% 
% Example:
%   dimVec(1)
% ans =
%   [1 0]
%
% Example:
%   dimVec(3)
% ans =
%   [0 0 1]
%
% Example:
%   dimVec(2, 7, 5, 1)
% ans =
%   [1 7 1 1 1]
%
%
% vec = dimVec(cDim, v, nDim, vOthers, true)
% 
% Example:
%   dimVec(2, 7, 5, [], true)
% ans =
%   {[], 7, [], [], []}
%
%
% See also REPMAT, RESHAPE, VECONDIM.

if nargin < 5 || isempty(cellOut), cellOut = false; end

if cellOut
    if isempty(nDim),                   nDim = max(cDim,2); end
    
    vec = repmat({vOthers}, [1 nDim]);
    vec(cDim) = {v};
    
else
    if nargin < 2 || isempty(v),        v = 1; end
    if nargin < 3 || isempty(nDim),     nDim = max(cDim, 2); end
    if nargin < 4 || isempty(vOthers),  vOthers = 0; end

    vec = vOthers * ones(1, nDim);
    vec(cDim) = v;
end