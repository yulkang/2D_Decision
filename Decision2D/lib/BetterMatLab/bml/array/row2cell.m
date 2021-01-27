function v = row2cell(v, onlyIfNotCell)
% v = row2cell(v, onlyIfNotCell=false)

if nargin < 2 || isempty(onlyIfNotCell), onlyIfNotCell = false; end
if onlyIfNotCell && iscell(v), return; end

if isempty(v)
    v = {v};
else
    r = size(v,1);
    c = size(v,2);
    v = mat2cell(v, ones(r,1), c);
end