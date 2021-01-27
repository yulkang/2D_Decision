function v = diff_same(v, dim)
% Symmetric numerical differentiation that returns the same-sized array.
%
% v = diff_same(v, dim)
%
% 2016 (c) Yul Kang. hk2699 at columbia dot edu.

if nargin < 2
    if isscalar(v)
        dim = 1;
    else
        dim = find(size(v) > 1, 1, 'first');
    end
end

n = size(v, dim);

if iscolumn(v) && dim == 1
    v = [v(2) - v(1); (v(3:n) - v(1:(n-2))) ./ 2; v(n) - v(n-1)];
    
elseif isrow(v) && dim == 2
    v = [v(2) - v(1), (v(3:n) - v(1:(n-2))) ./ 2, v(n) - v(n-1)];
    
elseif ismatrix(v) && ((dim == 1) || (dim == 2))
    if dim == 1
        v = [
            v(2,:) - v(1,:)
           (v(3:n,:) - v(1:(n-2),:)) ./ 2
            v(n,:) - v(n-1,:)
            ];
        
    elseif dim == 2
        v = [
            v(:,2) - v(:,1), ...
           (v(:,3:n) - v(:,1:(n-2))) ./ 2, ...
            v(:,n) - v(:,n-1)
            ];
    end
else
    % General but slow method for non-matrix or dim >= 3
    nd = ndims(v);

    C = repmat({':'}, [1, nd]);
    Cst1 = C;
    Cst1{dim} = 1;

    Cst2 = C;
    Cst2{dim} = 2;

    Cmid1 = C;
    Cmid1{dim} = 1:(n - 2);

    Cmid2 = C;
    Cmid2{dim} = 3:n;

    Cen1 = C;
    Cen1{dim} = n - 1;

    Cen2 = C;
    Cen2{dim} = n;

    v = cat(dim, ...
        v(Cst2{:}) - v(Cst1{:}), ...
       (v(Cmid2{:}) - v(Cmid1{:})) ./ 2, ...
        v(Cen2{:}) - v(Cen1{:}));
end
end