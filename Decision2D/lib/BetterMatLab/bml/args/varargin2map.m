function m = varargin2map(v)
assert(iscell(v) && size(v,2) == 2);
keys = v(:,1);
vals = v(:,2);
m = containers.Map(keys, vals, 'UniformValues', false);
end