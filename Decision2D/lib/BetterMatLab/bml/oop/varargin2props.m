function inst = varargin2props(inst, vararginCell, suppressError)
% USAGE: inst = varargin2props(inst, vararginCell, suppressError=false)
%
% For protected or private properties, tries set_PROP_NAME().
%
% 2015 (c) Yul Kang. yul dot kang dot on at gmail.

    if nargin < 2 || isempty(vararginCell)
        return; 
    end
    if nargin < 3
        suppressError = false; 
    end
    
    % Enforce behavior consistent with varargin2S and varargin2C
    if isstruct(vararginCell)
        vararginCell = varargin2C(vararginCell);
    elseif (size(vararginCell, 1) > 1) && (size(vararginCell, 2) == 2)
        vararginCell = hVec(vararginCell');
    else
        assert(iscell(vararginCell));
        assert(isrow(vararginCell));
    end

    % Assign one by one
    for iArgin = 1:2:numel(vararginCell)
        name = vararginCell{iArgin};
        
        inst = assign_field(inst, name, vararginCell{iArgin+1}, suppressError);
    end
end

function inst = assign_field(inst, name, v, suppressError)
    ix_period = find(name == '.');
    
    if any(ix_period)
        name_field = name(1:(ix_period - 1));
        name_rest  = name((ix_period + 1):end);
        
        try
            if bml.oop.isfield(inst, name_field)
                try
                    cv = inst.(name_field);
                catch
                    cv = inst.(['get_' name_field]);
                end
                
                cv = varargin2props(cv, ...
                    {name_rest, v}, suppressError);
                
                try
                    inst.(name_field) = cv;
                catch
                    inst.(['set_' name_field])(cv);
                end 

            elseif isobject(inst)
                inst.(['set_' name_field])(varargin2props(struct, ...
                    {name_rest, v}, suppressError));

            elseif isstruct(inst)
                inst.(name_field) = varargin2props(struct, ...
                    {name_rest, v}, suppressError);

            else
                error('Cannot parse subfield name!');
            end
        catch err
            if ~suppressError
                warning('Cannot parse subfield %s for class %s!\n', ...
                    name, class(inst));
            end
            rethrow(err);
        end
        return;
    end

    % Should use is_prop, etc., and figure out whether to throw error
    % or not.
    if isprop(inst, name)
        try
            inst.(name) = v;
        catch err
            if ismethod(inst, ['set_' name])
                inst.(['set_' name])(v);
            else
                if suppressError
                    warning(err_msg(err));
                else
                    rethrow(err);
                end
            end
        end
    elseif ~suppressError
        warning('No property named %s for class %s!\n', ...
            name, class(inst));
    end
end