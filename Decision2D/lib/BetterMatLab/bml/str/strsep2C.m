function C = strsep2C(s, sep)
% Similar to strsep2 but the output is one cell array
%
% C = strsep2C(s, sep='__')
%
% EXAMPLE:
% >> C = strsep2C('1+++b++3', '++')
% C = strsep2C('1+++b++3', '++')
% 
% >> C = strsep2C('1++++b++3', '++')
% C = strsep2C('1++++b++3', '++')
%
% See also: strsep, strsep2

if nargin < 2, sep = '__'; end

len = length(sep);
s   = [sep, s, sep];

ix   = strfind(s, sep);
n_ix = length(ix) - 1;

% Remove overlapping separaters
for ii = 2:n_ix
    if ix(ii) < ix(ii-1) + len
        ix(ii) = nan;
    end
end

ix   = ix(~isnan(ix));
n_ix = length(ix) - 1;

% Use the remainder
C = cell(1, n_ix);
for ii = 1:n_ix
    C{ii} = s((ix(ii) + len):(ix(ii+1) - 1));
end