classdef LogAxis < DeepCopyable
properties
    v_min0 = 0.1; % Relative to the distance between ticks
end
%% Init
methods
    function Ax = LogAxis(varargin)
        if nargin > 0
            Ax.init(varargin{:});
        end
    end
    function init(Ax, varargin)
        varargin2props(Ax, varargin);
    end
end
%% Convert
methods
    function v = convert_v(Ax, v0, tick0)
        tick = Ax.convert_tick(tick0);
        dtick = tick(2) - tick(1);
        v_min = dtick * Ax.v_min0 + tick(1);
        
        v = log(v0);
        v(v < v_min) = nan;
        
        assert(v0(1) == 0);
        v(1) = tick(1);

        if v0(3) < tick0(2)
            v(2) = tick(1) + dtick / 10;
            v(3) = nan;
        end
    end
    function [tick, ticklabel] = convert_tick(~, tick0)
        tick = log(tick0);
        dtick = tick(3) - tick(2);
        
        tick(tick0 == 0) = tick(2) - dtick;
        ticklabel = tick0;
    end
end
end