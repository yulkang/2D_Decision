function ds = setSs(ds, rows, Ss, varargin)
% ds = setSs(ds, rows, Ss, varargin)

%%
if ~exist('rows', 'var')
    rows = ':';
end
if ~iscell(Ss)
    Ss = num2cell(Ss);
end
Ss = Ss(:);

fs = cellfun(@(S) vVec(fieldnames(S)), Ss, 'UniformOutput', false);
fs = unique(cat(1, fs{:}), 'stable');

%%
nf = numel(fs);
for ii = 1:nf
    f = fs{ii};
    v = cellfun(@(S) S.(f), Ss, ...
        'UniformOutput', false, ...
        'ErrorHandler', @(varargin) []);
    ds.(f)(rows, 1) = v;
end