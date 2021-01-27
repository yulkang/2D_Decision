function v = sizes(mat, dim)
% v = sizes(mat, dim)

if ~exist('dim', 'var')
    v = size(mat);
else
    v = arrayfun(@(d) size(mat, d), dim);
end