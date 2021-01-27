function txt = wrap_text(txt0, varargin)
% txt = wrap_text(txt0, varargin)
% 'max_in_row', 40
% 'only_at', {}
S = varargin2S(varargin, {
    'max_in_row', 40
    });

len0 = numel(txt0);
n_row = ceil(len0 / S.max_in_row);
len1 = n_row * S.max_in_row;
txt = cell(1, n_row);
for ii = 1:n_row
    ix_st = (ii - 1) * S.max_in_row + 1;
    ix_en = min(ii * S.max_in_row, len0);
    txt{ii} = sprintf('%s\n', txt0(ix_st:ix_en));
end
if verLessThan('matlab', '8.6')
    txt = cellfun(@(s) s(1:(end-1)), txt, 'UniformOutput', false);
else
    txt = [txt{:}];
end
