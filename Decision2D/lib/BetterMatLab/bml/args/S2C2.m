function C = S2C2(S)
% Converts a struct or a cell array to [names(:), values(:)] cell array.

if isempty(S)
    C = cell(0,2);
    
elseif isstruct(S)
    C = S2C(S);
    C = reshape(C, 2, [])';
    
elseif iscell(S)
    C = S;
    if isvector(C)
        C = reshape(C, 2, [])';
    else
        assert(ismatrix(C));
        assert(size(C,2) == 2);
    end
else
    error('Give a struct or a cell array!');
end
end