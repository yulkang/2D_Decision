function v = struct2vec(S, fields)
% v = struct2vec(S, [fields])
C = struct2cell(S);

if nargin < 2
    v = cell2vec(C);
    return;
elseif ~iscell(fields)
    assert(ischar(fields));
    fields = {fields};
else
    assert(iscell(fields));
    assert(all(cellfun(@ischar, fields(:))));
end
fields0 = fieldnames(S);
[~,incl] = ismember(fields, fields0);
v = cell2mat(C(incl));