classdef EvidenceAxis < TimeAxis.TimeInheritable
    % EvAxis.EvidenceAxis

    % 2015 YK wrote the initial version.
    
properties
    max_drift = 0.5 * 100;
    max_bound = 3;
    ny = 2^9;
    min_sig = 1;
end
methods
    function [y, y0, dy, ny] = determine_y(Ev)
        [y, y0, dy, ny] = dtb.etc.determineY(Ev.max_drift, [-1, 1], ...
            Ev.dt, Ev.max_bound, -Ev.max_bound, Ev.ny, Ev.min_sig);
    end
    function v = get_y(Ev)
        v = Ev.determine_y;
    end
    function v = get_y0(Ev)
        [~, v] = Ev.determine_y;
    end
    
    % Violates Hollywood principle
%     function importWsDtb1D(Ev, W)    
%         Ev.Time = W.Time;
%         Ev.max_drift = max(W.Drift.get_drift_vec);
%         Ev.max_bound = max(W.Bound.get_bound_t_ch);
%     end
end
end