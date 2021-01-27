function c = S2C(s, f)
% Makes structure or object into a cell vector of variable names and values.
% If s is non-scalar, c is a cell array of the same size of cell vectors.
%
% c = S2C(s, [f])
% f: Fields to include.
%    {'field1', 'field2', ...} % If omitted, convert all.

if nargin < 2
    f = fieldnames(s);
end

if isscalar(s)
    if isstruct(s)
        c = hVec([f, struct2cell(s)]');

    else % slightly slower than S2C(struct(obj)) but doesn't produce warning.

        p = cellfun(@(cf) s.(cf), f, 'UniformOutput', false);
        c = hVec([f(:), p(:)]');
    end
else
    n = numel(s);
    c = cell(1, n);
    
    for ii = 1:n
        c{ii} = S2C(s(ii), f);
    end
    c = reshape(c, size(s));
end