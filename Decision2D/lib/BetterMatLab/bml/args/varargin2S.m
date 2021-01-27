function S = varargin2S(varCell, defaults, restrictInput) % , varargin)
% S = varargin2S({'argName1', arg1, 'argName2', arg2, ...}, [defaults = struct_or_cell, restrictInput = false])
% S = varargin2S({'argName1', arg1; 'argName2', arg2, ...}, ...)
% S = varargin2S({struct_or_dataset, 'argName1', arg1, ...}, ...)   
%
% Returns a struct like:
% S.argName1 == arg1
% S.argName2 == arg2
%
% S = varargin2S({{...}, {...}, ...}, ...) % Cell-in-cell format
% 
% Gives a struct array, matching the enclosing cell array's size.
%
% If restrictInput == 2, ignores inputs that are absent in defaults.
%
% Examples:
%
% Use
%   S = varargin2S(varargin, {'defaultArg1', defaultArg1, ...})
% in a function to create struct with the arguments as fields.
%
% Use
%   S = varargin2S(varargin, {'defaultArg1', defaultArg1, ...}, true)
% in a function to issue error when a variable name that's not in the defaults 
% is given in varargin.
% 
% See also: demoVarargin2S, varargin2V, varargin2C, arg, PsyLib
%
% 2013-2014 (c) Yul Kang. See help PsyLib for the license.

% if ~isempty(varargin)
%     opt = varargin2S(varargin);
% end

% To feed struct or dataset first
if ~iscell(varCell)
    varCell = {varCell};
end

% To use matrix format in the arguments, just use {...} in place of ...
if ~isempty(varCell) && iscell(varCell{1})
    varCell = varCell{1};
end

% Matrix format: {'name1', value1; ...}
if size(varCell,1) > 1
    varCell = varCell';
end

%% Restrict input
if nargin < 3, restrictInput = false; end % ~exist('restrictInput', 'var'), restrictInput = false; end

%% Defaults
if nargin < 2, % ~exist('defaults', 'var')
    defaults = struct;
    S = defaults;
elseif iscell(defaults)
    if isempty(defaults)
        S = struct;
    elseif isscalar(defaults)
        assert(isstruct(defaults{1}));
        S = defaults{1};
    else
        % If names are given in the 1st column and values in the 2nd column,
        if size(defaults,1) > 1
            defaults = defaults';
        end
        % Check if name-value pair
        if ~isNameValuePair(defaults)
            error('defaults must be argument name-value pairs!');
        end

        S = cell2struct(hVec(defaults(2:2:end)), hVec(defaults(1:2:end)), 2);
    end
elseif isstruct(defaults)
    S = defaults;
else
    % Convert dataset, struct, or object into a struct
    S = ds2struct(defaults);
end

%% Copy
if nargin < 1 || isempty(varCell)
    % Leaves S alone.
    
elseif iscell(varCell{1}) 
    % Cell in cell gives struct arrays
    siz = size(varCell);
    for ii = 1:numel(varCell)
        S = copyFields(S, ii, varargin2S(varCell{ii}, defaults));
    end
    if prod(siz) > 1, S = reshape(S, siz); end
    return;
    
elseif ~ischar(varCell{1}) 
    % Struct or obj
    S = copyFields(S, ds2struct(varCell{1}));

    if length(varCell) > 1
        S = varargin2S(varCell(2:end), S, restrictInput);
    end

    % Special case - dynStruct inputs return dynStruct.
    if isa(varCell{1}, 'dynStruct') && ~isa(S, 'dynStruct')
        S = dynStruct(S);
        S = copyFields(S, varCell{1}, varCell{1}.props_);
    end
else
    % Name-value pair
    fieldNames = varCell(1:2:end);
    ix_incl = 1:length(fieldNames);
    
    switch restrictInput
        case 1
            newArgs = setdiff(fieldNames, fieldnames(S)');
            if ~isempty(newArgs)
                error(['Unexpected argument:', sprintf(' %s', newArgs{:})]);
            end
        case 2
            [fieldNames, ix_incl] = intersect(fieldNames, fieldnames(S)');
    end

    for iField = ix_incl
        S.(fieldNames{iField}) = varCell{iField*2};
    end
end
