function ix = ix2py(ix, siz)
% Allows ix <= 0 similar to Python, except that 0 becomes the max, -k becomes max-k.
%
% ix = ix2py(ix, siz)
%
% ix  : a vector or a N x C matrix.
% siz : a C-vector.
%
% EXAMPLE:
% >> ix2py(-1:1, 5)
% ans =
%      4     5     1
%
% See also: IX_WRAP
%
% 2014 (c) Yul Kang. hk2699 at columbia dot edu.

if ischar(ix) && isequal(ix, ':')
    assert(nnz(siz > 1) <= 1, 'ix='':'' is allowed only for the size of vectors!');
    ix = 1:siz;
    return;
end

siz_ix = size(ix);

if isvector(ix), ix = ix(:); end

n_col = size(ix, 2);
for ii = 1:n_col
    ix(ix <= 0, ii) = max(ix(ix <= 0, ii) + siz(ii), 1);
end

ix = reshape(ix, siz_ix);
end