function varargout = ix2py(varargin)
% Allows ix <= 0 similar to Python, except that 0 becomes the max, -k becomes max-k.
%
% ix = ix2py(ix, siz)
%
% ix  : a vector or a N x C matrix.
% siz : a C-vector.
%
% EXAMPLE:
% >> ix2py(-1:1, 5)
% ans =
%      4     5     1
%
% See also: IX_WRAP
%
% 2014 (c) Yul Kang. hk2699 at columbia dot edu.
[varargout{1:nargout}] = ix2py(varargin{:});