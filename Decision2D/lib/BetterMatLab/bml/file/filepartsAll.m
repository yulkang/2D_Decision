function [p, n, e] = filepartsAll(f)
% [p, n, e] = filepartsAll(f)

sz = size(f);
p  = cell(sz);
n  = cell(sz);
e  = cell(sz);

for ii = 1:numel(f)
    [p{ii}, n{ii}, e{ii}] = fileparts(f{ii});
end
