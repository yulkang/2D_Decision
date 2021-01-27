function C = varargin2plot(C, defC)
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
    
if isempty(C)
    C = {};
elseif ~iscell(C)
    C = {C};
end
if ~isempty(C) && isLineSpec(C{1})
    C = [linespec2C(C{1}), hVec(C(2:end))];
end

if nargin < 2 || isempty(defC)
    defC = {};
elseif ~iscell(defC)
    defC = {defC};
end
if ~isempty(defC) && isLineSpec(defC{1})
    defC = [linespec2C(defC{1}), hVec(defC(2:end))];
end

C = varargin2C(C, defC);
    