function [ax_dst_new, h] = openfig_to_axes(file, ax_dst, varargin)
% [ax_dst_new, h] = openfig_to_axes(file, ax_dst, varargin)
%
% file: .fig file
% ax_dst: old axes
%
% h : struct containing handles
% ax_dst_new: the new, merged axes.

% 2016 Yul Kang. hk2699 at columbia dot edu.

S = varargin2S(varargin, {
    'ix_axes_to_load', ':'
    });

[~,~,ext] = fileparts(file);
if isempty(ext), file = [file, '.fig']; end
loadedfig = openfig(file, 'invisible');

ax_src = bml.plot.subplot_by_pos(loadedfig);
% ax_src = findobj(loadedfig, 'Type', 'Axes');
ax_src = ax_src(S.ix_axes_to_load);
n_ax = numel(ax_src);

for i_ax = n_ax:-1:1
    [ax_dst_new(i_ax), h(i_ax)] = ...
        bml.plot.copyaxes(ax_src(i_ax), ax_dst(i_ax));
end

h = reshape(h, size(ax_dst));
ax_dst_new = reshape(ax_dst_new, size(ax_dst));

delete(loadedfig);
end