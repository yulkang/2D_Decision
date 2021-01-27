function sub = ind2sub_mat(siz, ind)
% sub = ind2sub_mat(siz, ind)
%
% EXAMPLE:
% >> bml.matrix.ind2sub_mat([2 3], 1:6)
% ans =
%      1     1
%      2     1
%      1     2
%      2     2
%      1     3
%      2     3
     
assert(~isempty(siz));
assert(isnumeric(siz));
assert(isvector(siz));

assert(isnumeric(ind));
assert(isvector(ind));

n_dim = length(siz);

[subs{1:n_dim}] = ind2sub(siz, ind(:));
sub = cell2mat(subs);