function X = sums(X, dims, to_squeeze)
% X = sums(X, dims, [to_squeeze = false])
% sums(X) is equivalent to sum(X(:)) (convenient when X is sliced)

if nargin < 2
    X = sum(X(:));
    return;
end

for c_dim = dims
    X = sum(X, c_dim);
end

if (nargin >= 3) && to_squeeze
    X = squeeze(X);
end