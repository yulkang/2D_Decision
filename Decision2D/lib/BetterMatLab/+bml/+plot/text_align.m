function varargout = text_align(varargin)
% h = text_align(txt, varargin)
%
% OPTIONS
% -------
% 'corner', 'NW'
% 'margin', [0.1, 0.1]
% 'margin_unit', 'proportion'
% 'text_props', {}
% 'h_txt', [] % If given, update its position rather than creating new.
%
% 2015 (c) Yul Kang. hk2699 at columbia dot edu.
[varargout{1:nargout}] = text_align(varargin{:});