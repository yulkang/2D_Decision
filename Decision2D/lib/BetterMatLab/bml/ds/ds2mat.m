function v = ds2mat(ds)
% DS2MAT  convert a dataset to a matrix, assuming each column is a column vector.

col_names = ds.Properties.VarNames;

v = zeros(length(ds), length(col_names));

for i_var = 1:length(col_names)
    v(:, i_var) = ds.(col_names{i_var});
end