function ax = subplot_by_pos(fig, tol)
% Returns handles to axes in a matrix, arranged according to their posision.
% Useful when subplot() no longer works because plots were moved manually.
% ax(row, col) is the handle to the axes at that position.
%
% ax = subplot_by_pos(fig, [tol=1e-6])
%
% tol
% : tolerance when detecting vertical and horizonatl positions
%   as identical. Fed to uniquetol().

% Yul Kang (c) 2016. hk2699 at columbia dot edu.

if nargin < 1, fig = gcf; end
if nargin < 2, tol = 1e-3; end

%%
ax0 = findobj(fig, 'Type', 'Axes');
n = numel(ax0);
for ii = n:-1:1
    pos(ii,:) = get(ax0(ii), 'Position');
end

[~,~,icol] = uniquetol(pos(:,1), tol);
[~,~,irow] = uniquetol(pos(:,2), tol);

ncol = max(icol);
nrow = max(irow);
irow = nrow + 1 - irow;

ax = ghandles(nrow, ncol);
for ii = 1:n
    ax(irow(ii), icol(ii)) = ax0(ii);
end
end