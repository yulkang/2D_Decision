function res = indentStr(s, varargin)
% res = indentStr(s, varargin)
%
% 'len',      35
% 'indent',   17
% 'indent1st', false
%
% 2015 (c) Yul Kang. hk2699 at cumc dot columbia dot edu.

S = varargin2S(varargin, {
    'len',      50
    'indent',   17
    'indent1st', false
    });

ix  = [1:S.len:length(s), (length(s)+1)];
n   = length(ix)-1;
res = '';

for ii = 1:n
    if S.indent1st || (ii > 1)
        res = sprintf('%s%s', res, blanks(S.indent));
    end
    res = sprintf('%s%s\n', res, s(ix(ii):(ix(ii+1)-1)));
end