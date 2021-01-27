classdef TimeInheritable < DeepCopyable
    % TimeAxis.TimeInheritable
    
    % 2015 YK wrote the initial version.
    
properties (Access = private)
    % Always accessed by get_/set_Time.
    Time_
end
properties (Dependent)
    Time
    t
    dt
    nt
    max_t
end
methods
    function T = TimeInheritable(Time, time_args)
        % T = TimeInheritable(Time, time_args)
        T.add_deep_copy({'Time'});
        if nargin < 2, time_args = {}; end
        if nargin == 0 || isempty(Time)
            Time = TimeAxis.TimeRegularPositive(time_args{:});
        end
        T.set_Time(Time);
    end
    function v = get.t(T)
        v = T.get_t;
    end
    function v = get_t(T)
        v = T.Time.get_t;
    end
    function v = get.dt(T)
        v = T.get_dt;
    end
    function v = get_dt(T)
        v = T.Time.get_dt;
    end
    function v = get.nt(T)
        v = T.get_nt;
    end
    function v = get_nt(T)
        v = T.Time.get_nt;
    end
    function v = get.max_t(T)
        v = T.get_max_t;
    end
    function v = get_max_t(T)
        v = T.Time.get_max_t;
    end
    function set.t(T, v)
        T.set_t(v); % May be illegal for some Time objects
    end
    function set_t(T, v)
        T.Time.set_t(v); % May be illegal for some Time objects
    end
    function set.nt(T, v)
        T.set_nt(v); % May be illegal for some Time objects
    end
    function set_nt(T, v)
        T.Time.set_nt(v); % May be illegal for some Time objects
    end
    function set.dt(T, v)
        T.set_dt(v);
    end
    function set_dt(T, v)
        T.Time.set_dt(v);
    end
    function set.max_t(T, v)
        T.set_max_t = v;
    end
    function set_max_t(T, v)
        T.Time.set_max_t(v);
    end
    function set.Time(T, Time)
        T.set_Time(Time);
    end
    function set_Time(T, Time)
        T.set_Time_(Time);
    end
    function Time = get.Time(T)
        Time = T.get_Time;
    end
    function Time = get_Time(T)
        Time = T.get_Time_;
    end
end
methods (Sealed)
    function set_Time_(T, Time)
        T.Time_ = Time;
    end
    function Time = get_Time_(T)
        Time = T.Time_;
    end
end
end