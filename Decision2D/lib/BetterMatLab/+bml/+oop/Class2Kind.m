classdef Class2Kind < DeepCopyable
    % kind_general : the part of the class name following this string
    %                is considered the 'kind'. Set by set_kind_general().
%% kind
properties (Access = private)
    kind_general = '';
end
properties (Dependent)
    kind
end
properties
    kind_ = '';
end
methods
    function set_kind_general(W, v)
        W.kind_general = v;
    end
    function v = get_kind_general(W)
        v = W.kind_general;
    end
    function v = get.kind(W)
        v = W.get_kind;
    end
    function [v, v0] = get_kind(W)
        if isempty(W.kind_)
            v0 = class(W);
            if isempty(W.kind_general)
                v = v0;
            else
                ix = bml.str.strfind_end(v0, W.kind_general);
                assert(~isempty(ix));
                v = v0(ix:end);
            end
        else
            v = W.kind_;            
        end
    end
end
end