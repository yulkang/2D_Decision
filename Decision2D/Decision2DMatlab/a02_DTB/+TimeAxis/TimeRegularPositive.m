classdef TimeRegularPositive < matlab.mixin.Copyable
    % TimeAxis.TimeRegularPositive
    
    % 2015 YK wrote the initial version.

properties
    dt = 1 / 75; % 0.05;
    max_t = 5;
end
properties (Dependent)
    t
    nt
end
methods
    function Time = TimeRegularPositive(varargin)
        % Time = TimeRegularPositive(...)
        % 
        % Name-value pairs = defaults
        % dt = 0.05;
        % max_t = 5;
        
        if nargin > 0
            Time = varargin2fields(Time, varargin);
        end
    end
    %% Getters
    function v = get_dt(Time)
        v = Time.dt;
    end
    function v = get_max_t(Time)
        v = Time.max_t;
    end
    function v = get_t(Time)
        v = 0 + Time.dt * (0:(Time.nt-1));
    end
    function v = get_nt(Time)
        % Make sure max_t is included.
%         v = ceil(Time.max_t / Time.dt) + 1;
        v = Time.convert_sec2fr(Time.max_t); % Allow nt = 1
    end
    %% Conversion between fr and sec
    function fr_ix = convert_sec2fr_ix(Time, sec)
        fr_ix = min(max(Time.convert_sec2fr(sec), 1), Time.nt);
    end
    function fr = convert_sec2fr(Time, sec)
        fr = round(sec ./ Time.dt) + 1;
    end
    function sec = convert_fr2sec(Time, fr)
        sec = (fr - 1) .* Time.dt;
    end
    function v = get_refresh_rate(Time)
        v = 1 / Time.dt;
    end
    %% Setters
    function set_t(Time, t)
        assert(isvector(t) && t(1) == 0);
        Time.dt = t(2);
        Time.max_t = t(end);
    end
    function set_dt(Time, dt)
        assert(isscalar(dt) && dt > 0);
        Time.dt = dt;
    end
    function set_nt(Time, nt)
        assert(isscalar(nt) && nt >= 0);
        Time.max_t = Time.dt * (nt-1);
    end
    function set_max_t(Time, max_t)
        assert(isscalar(max_t) && max_t >= 0);
        Time.max_t = max_t;
    end
    %% Convenience functions
    function v = get.t(Time)
        v = Time.get_t();
    end
    function v = get.nt(Time)
        v = Time.get_nt();
    end
    function set.nt(Time, nt)
        Time.set_nt(nt);
    end
    function set.t(Time, t)
        Time.set_t(t);
    end
end
end