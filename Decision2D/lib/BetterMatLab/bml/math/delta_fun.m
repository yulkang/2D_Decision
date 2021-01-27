function tf = delta_fun(v, vec, min_val, normalize_dim)
% delta function, linearly interpolated if necessary.
%
% tf = delta_fun(v, vec, [min_val = 0, normalize = false])
%
% v   should be a numeric scalar or array within the range of vec.
% vec should be a monotonically increasing vector.
% 
% If v is an array, tf is the sum of the delta functions.
%
% Multiplying the step size (e.g., dt or dx) can be more accurate than 
% normalizing by sum.

if nargin < 3
    min_val = 0;
end
if nargin < 4
    normalize_dim = false;
end
if ~isscalar(v)
    tf = zeros(size(vec));
    nv = numel(v);
    
    for ii = 1:nv
        tf = tf + delta_fun(v(ii), vec, 0, false);
    end
    
else
    tf = zeros(size(vec));

    [~, ix] = min(abs(v - vec));
    w = v - vec(ix);

    if (w > 0) && (ix < length(vec))
        tf(ix + 1) = w / (vec(ix + 1) - vec(ix));
        tf(ix) = 1 - tf(ix + 1);

    elseif (w < 0) && (ix > 1)
        tf(ix - 1) = -w / (vec(ix) - vec(ix - 1));
        tf(ix) = 1 - tf(ix - 1);

    else
        tf(ix) = 1;
    end
end

if min_val ~= 0
    tf = max(tf, min_val);
end

if normalize_dim
    tf = bsxfun(@rdivide, tf, sum(tf, normalize_dim));
end
