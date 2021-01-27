classdef LinkProps < handle
%
% 2015 (c) Yul Kang. yul dot kang dot on at gmail dot com.
methods
    %% Linking properties
    function link_props(~)
        error('Not implemented! Modify in subclasses!');
        % No implementation here. Modify in subclasses. Always allow no-input calls.
    end
    function link_props_batch(me, props)
        % link_props_batch(me, props)
        % props: cell array of objects, property names, or both.
        for ii = 1:numel(props)
            prop = props{ii};
            if ischar(prop)
                me.(prop).link_props;
            else % The object is directly provided.
                prop.link_props;
            end
        end
    end
end
end