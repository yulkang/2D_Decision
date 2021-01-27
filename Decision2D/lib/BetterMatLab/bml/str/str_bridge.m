function s = str_bridge(connectChar, varargin)
% STR_BRIDGE  Bridge strings with connectChar except when it's empty
% : FULLSTR bridges even empty strings.
%
% s = str_bridge(connectChar, str1, str2, ...)
% s = str_bridge(connectChar, {str1, str2, ...})
%
% str: char or integer.
%
% Example:
% >> str_bridge('_', 'a', '', 'b', '')
% ans =
% a_b
% 
% See also: str_con, fullstr, funPrintfChop, funPrintf, funPrintfConnect, str, PsyLib
%
% 2013 (c) Yul Kang. See help PsyLib for the license.

if isempty(varargin)
    s = '';
    return;
end

if iscell(varargin{1})
    varargin = varargin{1};
end

% Convert integers into strings
for ii = 1:numel(varargin)
    if ~ischar(varargin{ii})
        varargin{ii} = sprintf('%g', varargin{ii});
    end
end
% varargin = cellfun(@(s) iif(ischar(s), s, true, sprintf('%g', s)), varargin, ...
%     'UniformOutput', false);

% Choose nonempty strings
ix = find(cellfun(@(s) ~isempty(s), varargin));
if isempty(ix)
    s = '';
    return;
else
    s = varargin{ix(1)};
    if length(ix) > 1
        s = [s sprintf([connectChar '%s'], varargin{ix(2:end)})];
    end
end