function c = varargin2C(varCell, defaults, restrict_input, n_pos_args, pos2default)
% C = varargin2C({'argName1', arg1, 'argName2', arg2, ...})
% C = varargin2C({struct_or_dataset, 'argName1', arg1, ...})
% C = varargin2C({pos_arg1, pos_arg2, ..., 'argName1', ...}, ...)
% C = varargin2C({...}, [defaults = struct_or_cell, restrictInput = false, n_pos_args = 0, pos2default=1])
% C = varargin2C({{...}, {...}, ...}, ...) % Cell of above cells
%
% n_pos_args: How many elements in front are positional arguments
% pos2default: 
%   0: any positional inputs are interpreted as is. 
%   1: empty input gives default value.
%   2: nan gives default. 
%   3: empty or nan gives default.
%
% Returns a cell like:
% C = {'argName1', arg1, 'argName2', arg2, ...}
%
% See also varargin2S, varargin2V

if nargin < 2, defaults = {}; end
if nargin < 3, restrict_input = false; end
if nargin < 4, n_pos_args = 0; end
if nargin < 5, pos2default = 1; end 

if n_pos_args == 0
    c = S2C(varargin2S(varCell, defaults, restrict_input));
    
else
    % Process positional arguments
    n_inp = min(length(varCell),  n_pos_args);
    n_def = min(length(defaults), n_pos_args);

    n_pos_exist = max(n_inp, n_def);

    if ~iscell(defaults)
        if ~isstruct(defaults)
            defaults = ds2struct(defaults);
        end
        defaults = S2C(defaults);
    end
    varCell((n_inp + 1):n_def) = defaults((n_inp + 1):n_def);
    
    switch pos2default
        case 1
            for ii = 1:n_inp
                if isempty(varCell{ii})
                    varCell{ii} = defaults{ii};
                end
            end
            
        case 2
            for ii = 1:n_inp
                if ismatrix(varCell{ii}) && isnan(varCell{ii})
                    varCell{ii} = defaults{ii};
                end
            end
            
        case 3
            for ii = 1:n_inp
                if isempty(varCell{ii}) || (ismatrix(varCell{ii}) && isnan(varCell{ii}))
                    varCell{ii} = defaults{ii};
                end
            end
    end
    
    % Process named arguments
    c = S2C(varargin2S( ...
            varCell( (n_pos_args + 1):end), ...
            defaults((n_pos_args + 1):end), ...
            restrict_input));
        
    % Merge
    c = [hVec(varCell(1:n_pos_exist)), c(:)'];
end

