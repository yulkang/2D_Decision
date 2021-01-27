function v = flips(v, dims)
% v = flips(v, dims)

if nargin < 2
    v = flip(v);
else
    for dim = 1:numel(dims)
        v = flip(v, dims(dim));
    end
end