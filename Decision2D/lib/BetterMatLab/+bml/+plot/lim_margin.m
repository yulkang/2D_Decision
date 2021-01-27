function lim_aft = lim_margin(varargin)
% lim_aft = lim_margin(varargin)
%
% OPTIONS:
% 'h', [] % handle to axes
% 'margin', 0.1
% 'obj_desc', {}
% 'direction', 'pos' % 'pos'|'neg'|'sym'|'bi'
% 'method', 'dif'
% 'fac', 1
% 'axis', 'y'
% 'min_v', []
% 'max_v', []

% 2016 (c) Yul Kang. hk2699 at columbia dot edu.

S = varargin2S(varargin, {
    'h', [] % handle to axes
    'margin', 0.1
    'obj_desc', {}
    'direction', 'pos' % 'pos'|'neg'|'sym'|'bi'
    'method', 'dif'
    'fac', 1
    'axis', 'y'
    'min_v', []
    'max_v', []
    });
if isempty(S.h)
    S.h = gca;
end

xy = bml.plot.get_all_xy(S.h, S.obj_desc);
v = xy(:, S.axis == 'xy');
switch S.method
    case 'dif'
        switch S.direction
            case 'pos'
                if isempty(S.min_v)
                    S.min_v = min(min(v(:)), 0);
                end
                if isempty(S.max_v)
                    S.max_v = max(max(v(:)), S.min_v + eps);
                end
                lim_aft = [ ...
                    S.min_v, ...
                    S.min_v + (S.max_v - S.min_v) * S.fac * (1 + S.margin)];
            case 'sym'
                if isempty(S.max_v)
                    S.max_v = max(max(abs(v(:))), eps);
                end
                lim_aft = S.max_v * S.fac * (1 + S.margin);
                lim_aft = [-lim_aft, lim_aft];
            otherwise
                error('Not implemented yet!');
        end
        axlim(S.h, S.axis, lim_aft);
        
    otherwise
        error('Not implemented yet!');
end
