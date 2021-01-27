function h = subplotRCs(nR, nC, r, c, varargin)
% h = subplotRCs(nR, nC, r, c, ...)
%
% When r or c is a cell array, return multiple h's in a matrix.
% Give [] to r and/or c to give num2cell(1:nR) and/or num2cell(1:nC).
%
% OPTIONS:
% 'clear',  false
% 'opt',    {}
%
% See also: subplotRC
%
% 2014 (c) Yul Kang. hk2699 at columbia dot edu.

S = varargin2S(varargin, {
    'clear', false
    'opt',   {}     % Options for subplotRC.
    });

if nargin < 3 || isempty(r), r = num2cell(1:nR); end
if nargin < 4 || isempty(c), c = num2cell(1:nC); end
if ~iscell(r), r = {r}; end
if ~iscell(c), c = {c}; end

r  = r(:);
c  = c(:);
nr = length(r);
nc = length(c);

if verLessThan('matlab', '8.4')
    h = zeros(nr, nc);
else
    h = gobjects(nr, nc);
end

for ir = 1:nr
    for ic = 1:nc
        cr = r{ir};
        cc = c{ic};
        h(ir, ic) = subplotRC(nR, nC, cr, cc, S.opt{:});
    end
end

if S.clear
    for ii = 1:numel(h)
        cla(h(ii));
    end
end
