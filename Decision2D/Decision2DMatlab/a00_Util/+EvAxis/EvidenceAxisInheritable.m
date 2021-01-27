classdef EvidenceAxisInheritable < DeepCopyable
    % Encapsulates EvAxis so that the info can be shared.
    %
    % EvAxis.EvidenceAxisInheritable

    % 2015 YK wrote the initial version.    

properties (Dependent)
    EvAxis
end
properties (Access = private)
    % EvAxis can be accessed only by calling set/get_EvAxis
    % outside this class.
    EvAxis_
end
properties (Dependent)
    % Get only
    y % Vector of y positions
    y0 % Vector y == 0
    
    % Get or set
    max_drift
    max_bound
    ny
end
methods
    function Ev = EvidenceAxisInheritable(EvAx, ev_args)
        Ev.add_deep_copy('EvAxis');
        if nargin < 2, ev_args = {}; end
        if nargin == 0 || isempty(EvAx)
            EvAx = EvAxis.EvidenceAxis(ev_args{:});
        end
        Ev.EvAxis_ = EvAx;
    end
end
%% Get/Set simple properties
methods
    function v = get_y(Ev)
        v = Ev.EvAxis.get_y();
    end
%     function set_y(Ev, v)
%         Ev.EvAxis.set_y(v);
%     end

    function v = get_y0(Ev)
        v = Ev.EvAxis.get_y0();
    end
%     function set_y0(Ev, v)
%         Ev.EvAxis.set_y0(v);
%     end    
    
    function v = get_max_drift(Ev)
        v = Ev.EvAxis.max_drift;
    end
    function set_max_drift(Ev, v)
        Ev.EvAxis.max_drift = v;
    end
    
    function set_max_bound(Ev, v)
        Ev.EvAxis.max_bound = v;
    end
    function v = get_max_bound(Ev)
        v = Ev.EvAxis.max_bound;
    end
    
    function set_ny(Ev, v)
        Ev.EvAxis.ny = v;
    end
    function v = get_ny(Ev)
        v = Ev.EvAxis.ny;
    end
end
%% Get/Set dependent properties
methods
    function v = get.y(Ev)
        v = Ev.get_y;
    end
%     function set.y(Ev, v)
%         Ev.set_y(v);
%     end

    function v = get.y0(Ev)
        v = Ev.get_y0;
    end
%     function set.y0(Ev, v)
%         Ev.set_y0(v);
%     end

    function v = get.max_drift(Ev)
        v = Ev.get_max_drift;
    end
    function set.max_drift(Ev, v)
        Ev.set_max_drift(v);
    end

    function v = get.max_bound(Ev)
        v = Ev.get_max_bound;
    end
    function set.max_bound(Ev, v)
        Ev.set_max_bound(v);
    end

    function v = get.ny(Ev)
        v = Ev.get_ny;
    end
    function set.ny(Ev, v)
        Ev.set_ny(v);
    end    
end
%% Utility
methods
    function [y, y0, dy, ny] = determine_y(Ev)
        [y, y0, dy, ny] = dtb.etc.determineY(Ev.max_drift, [-1, 1], ...
            Ev.dt, Ev.max_bound, -Ev.max_bound, Ev.ny);
    end
end
%% Get/Set object properties
methods
    function set.EvAxis(Ev, EvAxis)
        Ev.set_EvAxis(EvAxis);
    end
    function EvAxis = get.EvAxis(Ev)
        EvAxis = Ev.get_EvAxis;
    end
    function set_EvAxis(Ev, EvAxis)
        Ev.set_EvAxis_(EvAxis);
    end
    function EvAxis = get_EvAxis(Ev)
        EvAxis = Ev.get_EvAxis_;
    end
end
methods (Sealed)
    function set_EvAxis_(Ev, EvAxis)
        Ev.EvAxis_ = EvAxis;
    end
    function EvAxis = get_EvAxis_(Ev)
        EvAxis = Ev.EvAxis_;
    end
end
end