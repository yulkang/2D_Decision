function ds = cell2mat2_ds(ds, cols)
% ds = cell2mat2_ds(ds, [cols])

if nargin < 2 || isempty(cols)
    cols = ds.Properties.VarNames;
end
for col = cols(:)'
    v = ds.(col{1});
    
    if ~iscell(v)
        continue;
    end
    
    is_numeric = cellfun(@isnumeric, v);
    is_logical = cellfun(@islogical, v);
    is_char = cellfun(@ischar, v);
    is_empty = cellfun(@isempty, v);
    siz = cellfun(@size, v, 'UniformOutput', false);
    
    % Do not convert char
    if all(is_numeric)   
        ds.(col{1}) = cell2mat2(v);
    elseif all(is_logical)
        if all(cellfun(@isrow, v)) ...
                && all(cellfun(@(siz1) isequal(siz{1}, siz1), siz))
            ds.(col{1}) = cell2mat(v);
        end
    elseif all(is_char | is_empty)
        v(is_empty) = {''};
        ds.(col{1}) = v;
    end
end