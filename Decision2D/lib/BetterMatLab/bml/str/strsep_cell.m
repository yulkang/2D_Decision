function c = strsep_cell(s, sep, ix)
% c = strsep_cell(s, sep='_', ix=':')
%
% See also: strsep

if nargin < 2, sep = '_'; end
if nargin < 3, ix = ':'; end
n = nnz(s == sep) + 1;

[c{1:n}] = strsep(s, sep);

ix = ix2py(ix, n);
c = c(ix);
