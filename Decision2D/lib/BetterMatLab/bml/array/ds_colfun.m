function ds = ds_colfun(ds, fun, col)
% Apply a function to each column of a dataset.
%
% ds = ds_colfun(ds, fun, [col={'col1', ...}])

if nargin < 3, col = ds.Properties.VarNames; end

for c = col(:)'
    ds.(c{1}) = fun(ds.(c{1}));
end