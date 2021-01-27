function varargout = unique_general(varargin)
% Same as unique() but applies to any class. The order is always stable.
% 
% [v, ia, ic] = unique_general(v0)
%
% EXAMPLE:
% >> [c, ia, ic] = unique_general({2, 1, 4, 1, 'a', 'b', 'a', 4})
% c = 
%     [2]    [1]    [4]    'a'    'b'
% ia =
%      1     2     3     5     6
% ic =
%      1     2     3     2     4     5     4     3% 
%
% >> [c, ia, ic] = unique_general([2 1 2 4 3 4 1])
% c =
%      2     1     4     3
% ia =
%      1     2     4     5
% ic =
%      1     2     1     3     4     3     2
[varargout{1:nargout}] = unique_general(varargin{:});