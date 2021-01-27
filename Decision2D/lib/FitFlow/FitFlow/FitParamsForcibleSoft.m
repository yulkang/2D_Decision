classdef FitParamsForcibleSoft < FitParams
% FitParams with force-set th, accessed via get/set_th_(name).
% Does not affect existing code that uses th.
%
% 2015 (c) Yul Kang. hk2699 at cumc dot columbia dot edu.

properties (SetAccess = protected)
    th_forced = struct;
end
%% th_forced : Add/Override parameters
methods
    function add_th_forced(Params, name, v)
        Params.th_forced.(name) = v;
    end
    function remove_th_forced(Params, name)
        Params.th_forced = rmfield(Params.th_forced, name);
    end
end
%% Modify FitParams
methods
    function v = get_th_(Params, name)
        if isfield(Params.th_forced, name)
            v = Params.th_forced.(name);
        else
            v = Params.th.(name);
        end
    end
    function v = set_th_(Params, name, v)
        if isfield(Params.th_forced, name)
            Params.th_forced.(name) = v;
        else
            Params.th.(name) = v;
        end
    end
end
%% Test
methods (Static)
    function Params = test
        Params = copyprops(FitParamsForceTh, FitParams.test);
        
        %% Test th_forced : overriding existing th
        Params.add_th_forced('param1', 123);
        disp(Params.th.param1);
        assert(isequal(Params.th.param1, 123));
        
        %% Test th_forced : adding new th
        Params.add_th_forced('param_forced', 456);
        disp(Params.th.param_forced);
        assert(isequal(Params.th.param_forced, 456));
    end
end
end