function varargout = varargin2plot(varargin)
% C = varargin2plot(C, defaultC)
%
% EXAMPLES:
% varargin2plot({'r-'}, {'b*--'})
% ans = 
%     'LineStyle'    '-'    'Marker'    '*'    'Color'    'r'
% 
% varargin2plot({'r--'}, {'b*:'})
% ans = 
%     'LineStyle'    '--'    'Marker'    '*'    'Color'    'r'
% 
% varargin2plot({'--'}, {'b*:'})
% ans = 
%     'LineStyle'    '--'    'Marker'    '*'    'Color'    'b'
% 
% varargin2plot({'--'}, {'b:', 'Marker', '*'})
% ans = 
%     'LineStyle'    '--'    'Color'    'b'    'Marker'    '*'
% 
% varargin2plot({'--+'}, {'b:', 'Marker', '*'})
% ans = 
%     'LineStyle'    '--'    'Color'    'b'    'Marker'    '+'
% 
% varargin2plot({'+', 'LineStyle', '--'}, {'b:', 'Marker', '*'})
% ans = 
%     'LineStyle'    '--'    'Color'    'b'    'Marker'    '+'

% 2015 (c) Yul Kang. hk2699 at columbia dot edu.

[varargout{1:nargout}] = varargin2plot(varargin{:});