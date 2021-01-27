function res = bsxFind(v, rep)
% BSXFIND Find match and returns index.
% 
% res = bsxFind(v, rep)
%
% v should be a vector.
% rep should be a vector, and have non-repeating elements.
%
% res has the same size as v.
%
% Example:
%
% bsxFind([1 2 3 4]', [1 2 4])
% ans =
%      1
%      2
%      0
%      3
%
% See also: bsxEq, find, bsxfun, strcmp, data, PsyLib
%
% 2013 (c) Yul Kang. See help PsyLib for the license.

[r c] = find(bsxfun(@(a,b) eq(a,b), v(:), rep(:)'));

res = zeros(size(v));
res(r) = c;
end