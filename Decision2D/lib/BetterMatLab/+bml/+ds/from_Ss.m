function ds = from_Ss(Ss, varargin)
% ds = from_Ss(Ss, varargin)

Ss = Ss(:);
n = numel(Ss);
ds = bml.ds.setSs(dataset, 1:n, Ss, varargin{:});
end