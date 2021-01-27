function [merged, h] = copyaxes(src, dst, varargin)
% [merged, h] = copyaxes(src, dst, ...)
%
% src, dst : handles of the axes.
%
% h.src, h.dst : struct containing handles, e.g.,
% h.src.line, text, legend, and children.
%
% merged : the new, merged axes.
%
% OPTIONS:
% 'copy_legend', true % whether to copy legends

% 2016 Yul Kang. hk2699 at columbia dot edu.

h.src.axes = src;
h.dst.axes = dst;

is_dst_empty = isempty(get(dst, 'Children'));
xlim_dst = xlim(dst);
ylim_dst = ylim(dst);
xlim_src = xlim(src);
ylim_src = ylim(src);

h.src.peers = getappdata(h.src.axes, 'LayoutPeers');
fig_dst = get(h.dst.axes, 'Parent');

if isempty(h.src.peers)
    h.src.axes = copyobj(h.src.axes, fig_dst);
else
    h.dst.peers = copyobj([h.src.axes; h.src.peers(:)], fig_dst);
    h.src.axes = h.dst.peers(1);
    h.dst.peers = h.dst.peers(2:end);
end

set(h.src.axes, 'Position', get(h.dst.axes, 'Position'));
h.src = copyFields(h.src, bml.plot.figure2struct(h.src.axes));

if ~is_dst_empty
    xlim(h.src.axes, ...
        [min(xlim_dst(1), xlim_src(1)), max(xlim_dst(2), xlim_src(2))]);
    ylim(h.src.axes, ...
        [min(ylim_dst(1), ylim_src(1)), max(ylim_dst(2), ylim_src(2))]);
end

h.dst.children = copyobj(get(h.dst.axes, 'Children'), h.src.axes);
h.dst = copyFields(h.dst, bml.plot.figure2struct(h.dst.children));

delete(dst);
merged = h.src.axes; % In this implementation, the old axes is replaced.
h.dst.axes = merged;

% % Another implementation that preserves dst but does not work

% h.src.children = copyobj(get(h.src.axes, 'Children'), h.dst.axes);
% h.src = copyFields(h.src, bml.plot.figure2struct(h.src.children));
% 
% h.src.axes = src;
% h.dst.axes = dst;
% 
% is_dst_empty = isempty(get(dst, 'Children'));
% xlim_dst = xlim(dst);
% ylim_dst = ylim(dst);
% xlim_src = xlim(src);
% ylim_src = ylim(src);

% bml.oop.copyprops(dst, src, 'props_to_skip', {
%         'Parent', 'Children', ...
%         'Position', 'OuterPosition'}, ...
%     'hide_error', true);

% if ~is_dst_empty
%     xlim([min(xlim_dst(1), xlim_src(1)), max(xlim_dst(2), xlim_src(2))]);
%     ylim([min(ylim_dst(1), ylim_src(1)), max(ylim_dst(2), ylim_src(2))]);
% end

% for ax = {'src', 'dst'}
%     h.(ax{1}) = copyFields(h.(ax{1}), bml.plot.figure2struct(h.(ax{1}).axes));
% end
% 
% delete(h.dst.axes);
end