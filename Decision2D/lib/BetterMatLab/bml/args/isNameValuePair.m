function tf = isNameValuePair(c)
% isNameValuePair  Examines if the argument is in the format of {'name1', value1, ...}
%
% See also: varargin2S, varargin2V, arg, PsyLib
%
% 2013 (c) Yul Kang. See help PsyLib for the license.

    tf = iscell(c) && (mod(numel(c),2) == 0) && all(cellfun(@ischar, c(1:2:end)));
end