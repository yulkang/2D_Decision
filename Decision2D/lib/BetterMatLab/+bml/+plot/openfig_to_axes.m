function varargout = openfig_to_axes(varargin)
% [ax_dst_new, h] = openfig_to_axes(file, ax_dst, varargin)
%
% file: .fig file
% ax_dst: old axes
%
% h : struct containing handles
% ax_dst_new: the new, merged axes.
[varargout{1:nargout}] = openfig_to_axes(varargin{:});