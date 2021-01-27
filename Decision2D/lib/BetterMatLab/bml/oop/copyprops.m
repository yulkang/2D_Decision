function [dst, info] = copyprops(dst, src, varargin)
% [dst, info] = copyprops(dst, src, ...)
%
% src, dst : struct or an object
%
% info.handles.(prop) : handle properties, when skip_handle = true
%
% OPTIONS: all applies to the source (src) side.
% -------
% 'props', [] % Give names {'prop1', ...}, [] (all except skipped), or {} (none).
% 'props_to_skip', {}
% 'only_public', false % skip_internal and skip_protected
% 'skip_absent', true
% 'skip_dependent', true
% 'skip_transient', true
% 'skip_hidden', false
% 'skip_protected', false
% 'skip_error', true
% 'skip_handle', false
% 'skip_internal', false % skip properties with a name that end with '_'
% 'hide_error', false

% 2015 (c) Yul Kang. yul dot kang dot on at gmail.

S = varargin2S(varargin, {
    'props', [] % Give names {'prop1', ...}, [] (all except skipped), or {} (none).
    'props_to_skip', {}
    'only_public', false % skip_internal and skip_protected
    'skip_absent', true
    'skip_dependent', false
    'skip_transient', false
    'skip_hidden', false
    'skip_protected', false
    'skip_error', true
    'skip_handle', false
    'skip_internal', false % skip properties with a name that end with '_'
    'hide_error', false
    });
info = struct;

if S.only_public
    S.skip_internal = true;
    S.skip_protected = true;
end

if S.hide_error
    S.skip_error = true;
end

% Source props
isobject_src = isobject(src) && ~istable(src) ...
    && ~isa(src, 'dataset');

props_src_name = S.props;

if ~isobject_src
    if isequal(props_src_name, [])
        props_src_name = fieldnames(src);
    end
else
    mc = metaclass(src);
    props_src = mc.PropertyList;

    if isequal(props_src_name, [])
        incl = true(numel(props_src), 1);

        if S.skip_dependent
            incl = incl & ~vVec([props_src.Dependent]);
        end
        if S.skip_transient
            incl = incl & ~vVec([props_src.Transient]);
        end
        if S.skip_hidden
            incl = incl & ~vVec([props_src.Hidden]);
        end
        if S.skip_protected
            incl = incl & ~vVec(strcmp('public', {props_src.GetAccess}));
        end
        props_src = props_src(incl);
        props_src_name = {props_src.Name};

        if S.skip_internal
            is_internal = cellfun(@(s) s(end) == '_', props_src_name);
            props_src_name = props_src_name(~is_internal);
        end
    end
end

props_src_name = setdiff(props_src_name, S.props_to_skip, 'stable');

assert(iscell(props_src_name));
assert(all(cellfun(@ischar, props_src_name(:))));

% Destination props
isobject_dst = isobject(dst) && ~istable(dst) ...
    && ~isa(dst, 'dataset');
if isobject_dst
    mc_dst = metaclass(dst);
    props_dst = mc_dst.PropertyList;
    props_dst_name = {props_dst.Name};
end

% Copy
n = numel(props_src_name);
for ii = 1:n
    prop_name = props_src_name{ii};
    
    % From src to v
    if S.skip_absent && ~bml.oop.isfield(src, prop_name)
        continue;
    end
    
    try
        if isobject_src
            prop_src = props_src(strcmp(props_src_name, prop_name));
            if isprop(src, prop_name) ...
                    && strcmp(prop_src.GetAccess, 'public')
                v = src.(prop_name);
            else
                v = src.(['get_' prop_name]);
            end
        else
            v = src.(prop_name);
        end
    catch err
        if S.skip_error
            if ~S.hide_error
                warning(err_msg(err));
            end
            continue;
        else
            rethrow(err);
        end
    end
    
    if S.skip_handle && isa(v, 'handle')
        info.handles.(prop_name) = v;
        continue;
    end
    
    % From v to dst
    if isobject_dst
        prop_dst = props_dst(strcmp(prop_name, props_dst_name));
    end
    
    try
        if ~isobject_dst ...
                || isequal(isprop(dst, prop_name), true)
            
            if isobject_dst
                if prop_dst.Dependent ...
                        && ~ismethod(dst, ['set.' prop_name])
                    continue;
                elseif ~strcmp(prop_dst.SetAccess, 'public')
                    if S.skip_protected
                        continue;
                    elseif ismethod(dst, ['set_' prop_name])
                        dst.(['set_' prop_name])(v);
                    else
                        error('Method %s.set_%s does not exist!\n', ...
                            class(dst), prop_name);
                    end
                end
            end
            dst.(prop_name) = v;
            
        elseif ~S.skip_absent
            error('Class %s does not have a proprty %s!\n', ...
                class(dst), prop_name);
        end
    catch err
        if S.skip_error
            if ~S.hide_error
                warning('Error while setting %s.%s:\n', ...
                    class(dst), prop_name);
                warning(err_msg(err));
            end
        else
            rethrow(err);
        end
    end
end