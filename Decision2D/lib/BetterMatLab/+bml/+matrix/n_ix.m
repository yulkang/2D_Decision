function n = n_ix(ix, n_all)
% Number of elements for either numeric or logical indices.
%
% n = n_ix(ix, n_all)
%
% n_all: required only for the case where ix = ':'

assert(isvector(ix));
if isnumeric(ix)
    n = length(ix);
elseif islogical(ix)
    n = nnz(ix);
elseif ischar(ix) && isequal(ix, ':')
    n = n_all; % n_all is required in this case.
else
    error('ix must be either numeric or logical!');
end