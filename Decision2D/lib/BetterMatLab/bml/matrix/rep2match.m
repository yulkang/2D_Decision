function varargout = rep2match(inp, varargin)
% varargout = rep2match(inp, varargin)
%
% Return arrays that are replicated to match the maximum size on each dimension.
%
% inp{k}: k-th array.
%
% See also: rep2fit, plot3.
%
% 2013 (c) Yul Kang, hk2699 at columbia dot edu.

S = varargin2S(varargin, {
    'dim', ':' % give specific dims to match
    });

n        = numel(inp);
ndimsAll = cellfun(@ndims, inp);
ndimsMax = max(ndimsAll);
if ischar(S.dim) && isequal(S.dim, ':')
    S.dim = 1:ndimsMax;
end

siz      = zeros(1,ndimsMax);

for ii = 1:n
    cSiz = size(inp{ii});
    siz(S.dim)  = max(siz(S.dim), [cSiz(S.dim), ones(1,ndimsMax - ndimsAll(ii))]);
end

varargout = cell(1,n);
for ii = 1:n
    varargout{ii} = bml.matrix.rep2fit(inp{ii}, siz);
end