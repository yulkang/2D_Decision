function tf = inputYN_def(msg, tf_default, varargin)
% tf = inputYN_def(msg, tf_default, sprintf_args)
%
% See also inputYN, misc, PsyLib
%
% 2013 (c) Yul Kang. See help PsyLib for the license.

% Use msg as format
if ~isempty(varargin)
    msg = sprintf(msg, varargin{:});
end

% Capitalize depending on the default
if tf_default == true
    msg = [msg, ' (Y/n)? '];
else
    msg = [msg, ' (y/N)? '];
end

% Get response
resp = input(msg, 's');

% Set to default if empty
if isempty(resp), tf = tf_default; return; end

% Get response until valid
while ~strcmpi(resp, 'y') && ~strcmpi(resp, 'n')
    resp = input(msg, 's');
    
    if isempty(resp), tf = tf_default; return; end
end

% Set to response
tf = strcmpi(resp, 'y');