function symmetric_y(varargin)
if ~isempty(varargin)
    ax = varargin{1};
else
    ax = gca;
end

for i=1:length(ax)
    yl = get(ax(i),'ylim');
    lim = max(abs(yl));
    set(ax(i),'ylim',[-lim lim]);
end
