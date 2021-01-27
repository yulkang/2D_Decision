function hold(ax, arg)
% hold(ax, arg)
%
% Same as MATLAB's hold except that this allows ax to be an array.
if ischar(ax)
    arg = ax;
    ax = gca;
end
for ii = 1:numel(ax)
    hold(ax(ii), arg);
end