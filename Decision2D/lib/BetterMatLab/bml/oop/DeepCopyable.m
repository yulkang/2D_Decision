classdef DeepCopyable < matlab.mixin.Copyable
% DeepCopyable enables deep_copy for subclasses.
%
% Add properties to deep_copy with add_deep_copy({name1, name2, ...}),
% preferably on construction.
%
% If the property is a struct, its fields are deep copied.
% If the property is a cell, its elements are deep copied.
% 
% You need to add only DeepCopyable or matlab.mixin.Copyable properties:
% value properties are copied by value anyway (without add_deep_copy), and
% handle properties cannot be copied (make them subclass matlab.mixin.Copyable).
% 
% When adding protected or private properties, 
% provide get_PROPERTYNAME and/or set_PROPERTYNAME without additional arguments.
% The properties are set in the order given in add_deep_copy.
%
% TODO: Implement when necessary:
% dc.deep_copy_first(prop)
% dc.deep_copy_last(prop)
% dc.deep_copy_A_before_B(prop_A, prop_B) 
% dc.deep_copy_B_after_A(prop_A, prop_B) 
%   Puts prop immediately after prop1 if prop1 was after prop2.
%   Ignored if prop1 was before prop2.
%
% See also: DeepCopyableTest
%
% 2015 (c) Yul Kang. hk2699 at cumc dot columbia dot edu.
properties (Access = private)
    deep_copy_props_ = {}; % Names of properties to deep_copy. Can be struct or cell.
end
%% Core functions
methods
    function add_deep_copy(dc, names)
        % add_deep_copy(dc, names)
        if ischar(names), names = {names}; end
        assert(all(cellfun(@ischar, names)));        
        dc.deep_copy_props_ = union(dc.deep_copy_props_, names, 'stable');
    end
    function remove_deep_copy(dc, names)
        % remove_deep_copy(dc, names)
        if ischar(names), names = {names}; end
        assert(all(cellfun(@ischar, names)));
        dc.deep_copy_props_ = setdiff(dc.deep_copy_props_, names, 'stable');
    end
    function [dc2, copied, already_copied] = deep_copy(dc, copied)
        % [dc2, copied, already_copied] = deep_copy(dc, copied={})
        %
        % copied : used internally to keep handles already copied, and
        %          prevent copying them again.

        % Initialize copied.
        if nargin < 2
            copied = cell(0,2); 
        end
        
        % Shallow copy itself
        [dc2, copied, already_copied] = dc.shallow_copy_handle(dc, copied);
        if already_copied, return; end
        
        % Deep copy properties
        if ~isscalar(dc)
            for ii = 1:numel(dc)
                [dc2(ii), copied] = ...
                    dc(ii).copy_properties(dc(ii), dc2(ii), dc(ii).deep_copy_props_, copied);
            end
        else
            [dc2, copied] = dc.copy_properties(dc, dc2, dc.deep_copy_props_, copied);
        end
    end
    function names = get_deep_copy(dc)
        % Mostly for testing purposes
        names = dc.deep_copy_props_;
    end
end
methods (Static)
    function [dst, copied] = copy_properties(src, dst, prop_names, copied)
        % [dst, copied] = copy_properties(src, dst, prop_names, copied)
        for prop_name = prop_names(:)'
            copied = DeepCopyable.copy_prop(src, dst, prop_name{1}, copied);
        end        
    end
    function copied = copy_prop(src, dst, prop_name, copied)
        % copied = copy_prop(src, dst, prop_name, copied)
        %
        % Since the property is supposed to be a handle or 
        % struct or cell arrays containing handles,
        % assigning it shouldn't be a heavyweight operation.
        
        % Get the source property
        try
            prop = src.(prop_name);
        catch
            try
                prop = src.(['get_' prop_name]);
            catch
                error([class(src) '.get_' prop_name '() is not defined!']);
            end
        end
        
        % Copy to temp
        if isstruct(prop)
            temp = struct;
            for f = fieldnames(prop)'
                [temp.(f{1}), copied] = DeepCopyable.deep_copy_handle(prop.(f{1}), copied);
            end
        elseif iscell(prop)
            temp = cell(size(prop));
            for ii = 1:numel(prop)
                [temp{ii}, copied] = DeepCopyable.deep_copy_handle(prop{ii}, copied);
            end
        else
            [temp, copied] = DeepCopyable.deep_copy_handle(prop, copied);
        end
        
        % Set to dst property
        try
            dst.(prop_name) = temp;
        catch
            set_method = ['set_' prop_name];
            if ismethod(class(dst), set_method)
                dst.(set_method)(temp);
            else
                error([class(dst) '.' set_method '() is not defined!']);
            end
        end
    end
    function [h, copied] = deep_copy_handle(h, copied)
        % If already copied, return the cached copy.
        %
        % [dst, copied, already_copied] = deep_copy_handle(src, copied)
        %     copied{k,1}: original 
        %     copied{k,2}: copy
        
        % Copy
        if isa(h, 'DeepCopyable')
            [h, copied] = deep_copy(h, copied);
        elseif isa(h, 'matlab.mixin.Copyable')
            [h, copied] = DeepCopyable.shallow_copy_handle(h, copied);
        elseif isa(h, 'handle')
            error('Properties to deep_copy must be DeepCopyable or matlab.mixin.Copyable!');
        else
            % If src is not a handle, return it as is.
        end        
    end 
    function [h, copied, already_copied] = shallow_copy_handle(h, copied)
        % If already copied, return the cached copy.
        %
        % [dst, copied] = shallow_copy_handle(src, copied)
        %     copied{k,1}: original 
        %     copied{k,2}: copy
        
        already_copied = false;
        assert(isa(h, 'matlab.mixin.Copyable'));
        
        % When DC matches copied{k,1} for some k,
        % the copied{k,2} is returned.
        % Useful for copying circular references without infinite recursion.
        for ii = 1:size(copied,1)
            % DEBUG^2: avoids errors from size mismatch, etc.
            % DEBUG: replacing h == copied{ii,1}
            if isequal(h, copied{ii,1}) 
                % Check again if the same handle.
                % This is NOT equivalent to isequal!
                if h == copied{ii,1} 
                    h = copied{ii,2};
                    already_copied = true;
                    return;
                end
            end
        end

        % Update original
        copied{end+1,1} = h;

        % Copy
        h = copy(h);

        % Update copied
        copied{end,  2} = h;
    end 
end
end