function [hch_out, pos] = shiftpos(h, ch, shift)
% Shift self or children's position.
%
% [hch_out, pos] = shiftpos(h, shift)
% [hch_out, pos] = shiftpos(h, ch, shift)
%
% h    : Handle(s) of figure or axis.
% ch   : Child's name, like 'XLabel', 'YLabel', 'Title'.
% shift: [xshift, yshift] or [xshift, yshift, zshift].
%
% hch  : Handle of children.
% pos  : Shifted position.
%
% EXAMPLE:
% >> shiftpos(gcf, 'Children', [0.05, -0.05])
% % Useful for shifting axes positions for printing.
% % The shift is relative to the position before ANY shiftpos is applied. 
% % That is, even when shiftpos is repeated, it does not move the axes further, 
% % when the amount of shift is the same.
%
% 2014 (c) Yul Kang. hk2699 at columbia dot edu.

% Determine hpr and hch
if nargin < 3
    shift = ch;
    hch   = num2cell(h);
    hpr   = get(h(1), 'Parent');
else
    hpr   = h; % Parent
    hch   = get(h, ch);
    if ~iscell(hch), hch = num2cell(hch); end
end
n = numel(hch);

% Get position
for ii = 1:n
    pos(ii,:) = get(hch{ii}, 'Position'); %#ok<AGROW>
end
    
% Fill shift
nc = size(pos,2);
if size(shift, 2) < nc, shift(end, nc) = 0; end

% Determine previous shift
pshift = nan(1,4);
if isappdata(hpr, 'shiftpos')
    prev    = getappdata(hpr, 'shiftpos');
    nprev   = length(prev);
    to_delete = false(1, nprev);
    
    for ii = 1:nprev
        if isequal(prev(ii).hch, hch)
            pshift = prev(ii).shift;
            prev(ii).shift = shift;
            setappdata(hpr, 'shiftpos', prev);
        elseif any(~cellfun(@isvalidhandle, prev(ii).hch))
            to_delete(ii) = true;
        end
    end
    prev  = prev(~to_delete);
    nprev = length(prev);
    
    if any(isnan(pshift))
        nprev = nprev + 1;
        prev(nprev).hch     = hch;
        prev(nprev).shift   = shift;
        setappdata(hpr, 'shiftpos', prev);
        pshift = zeros(1,4);
    end
else
    prev.hch    = hch;
    prev.shift  = shift;
    setappdata(hpr, 'shiftpos', prev);
    pshift = zeros(1,4);
end

% Shift
pos = bsxfun(@plus, pos, bsxfun(@minus, shift, pshift));

% Set position
for ii = 1:numel(hch)
    set(hch{ii}, 'Position', pos(ii,:));
end

% Output
if nargout >= 1, hch_out = hch; end