function X = means(X, dims, to_squeeze)
% X = means(X, dims, [to_squeeze = false])
% means(X) is equivalent to mean(X(:)) (convenient when X is sliced)

if nargin < 2
    X = mean(X(:));
    return;
end

for c_dim = dims
    X = mean(X, c_dim);
end

if (nargin >= 3) && to_squeeze
    X = squeeze(X);
end