function tf = bsxEq(a, b)
% tf = bsxEq(a, b)
%
% tf : logical column vector of length(a) of whether each element of a
%      equals any of b.

tf = logical(any(bsxfun(@eq, vVec(a), hVec(b)), 2));
end