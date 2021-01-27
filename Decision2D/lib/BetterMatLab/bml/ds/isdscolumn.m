function tf = isdscolumn(ds, col)
% True if COL is a column of DS. 
% COL can be either a string or a cell array of strings.
%
% tf = isdscolumn(ds, col)

tf = ismember(col, ds.Properties.VarNames);