function beautify(ax)
% beautify(ax=gca)
if ~exist('ax', 'var'), ax = gca; end

set(ax, 'Box', 'off', 'TickDir', 'out');
end