function varargout = dealDef(C, default_values, empty_if_no_default)
% Same as deal() but gives defaults if the cell array is shorter than the number of outputs.
%
% [out1, out2, ...] = dealDef({in1, in2, ...}, {default1, default2, ...}, ...
%                             [empty_if_no_default = false])
%
% If number of default is less than the number of outputs, the output gets [].
% If empty2d is true, empty inputs are replcaed with the default.
%
% 2014 (c) Yul Kang. hk2699 at columbia dot edu.

varargout = cell(1, nargout);
nd = length(default_values);

if nargin < 3, empty_if_no_default = false; end

% Copy inputs
for ii = 1:length(C)
    varargout{ii} = C{ii};
    
    % If empty2d is true, empty inputs are replcaed with the default.
    if empty_if_no_default && isempty(varargout{ii}) && (ii <= nd)
        varargout{ii} = default_values{ii};
    end
end

% Copy defaults
for ii = (length(C)+1):min(nargout, nd)
    varargout{ii} = default_values{ii};
end