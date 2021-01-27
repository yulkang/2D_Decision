function [common_a, common_b] = intersect_ix_general(a, b, is_rows)
% Internal function used by (set_operation)_general functions.
% [common_a, common_b] = intersect_ix_general(a, b, is_rows)
%
% is_rows: logical scalar. Whether to compare rows or elements.
% common_a, b : logical vector.
%
% See also: union_general, setdiff_general
%
% 2015 (c) Yul Kang. hk2699 at cumc dot columbia dot edu.
if is_rows
    if isempty(a) || isempty(b)
        common_a = [];
        common_b = [];
        return;
    end
    
    assert(ismatrix(a) && ismatrix(b) && size(a,2) == size(b, 2));

    na = size(a,1);
    nb = size(b,1);
    common_a = false(na, 1);
    common_b = false(nb, 1);
    for ia = 1:na
        for ib = 1:nb
            if isequal(a(ia,:), b(ib,:))
                common_a(ia) = true;
                common_b(ib) = true;
                break;
            end
        end
    end        
else
    na = numel(a); % different from rows
    nb = numel(b); % different from rows
    common_a = false(na, 1);
    common_b = false(nb, 1);
    for ia = 1:na
        for ib = 1:nb
            if isequal(a(ia), b(ib)) % Different from rows
                common_a(ia) = true;
                common_b(ib) = true;
                break;
            end
        end
    end        
end
