function s = str_con(varargin)
% str_con  Connect nonempty strings with underscore.
%
% str_con(str1, str2, ...)
% str_con({str1, str2, ...})
%
% str: char or integer.
%
% See also str_bridge

s = str_bridge('_', varargin{:});
end