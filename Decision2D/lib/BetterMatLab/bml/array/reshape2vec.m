function dst = reshape2vec(src, dim)
% RESHAPE2VEC   Reshape an array into a vector on the given dimension.
%
% Example: 
%   reshape2vec(zeros(2,2), 2)
% ans =
%   [0 0 0 0]
%
% See also RESHAPE, DIMVEC, VECONDIM.

dimV    = vecOnDim(dim);
dst     = reshape(src, dimV{:});
end