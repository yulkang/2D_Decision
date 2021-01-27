function beautify_tick(ax, xy, varargin)
% beautify_tick(ax, xy, varargin)
%
% ax: axes
% xy: 'X' or 'Y'
% 
% OPTIONS:
% 'tick', { % Candidate sets of ticks in progressively smaller steps.
%     0:1:5
%     0:0.5:5
%     0:0.2:5
%     0:0.1:5
%     0:0.05:5
%     0:0.02:5
%     0:0.01:5
%     0:0.001:5
%     }
S = varargin2S(varargin, {
    'tick', { % Candidate sets of ticks in progressively smaller steps.
        0:1:5
        0:0.5:5
        0:0.2:5
        0:0.1:5
        0:0.05:5
        0:0.02:5
        0:0.01:5
        0:0.001:5
        }
    'min_n_tick', 2
    });

xy = upper(xy);
lim = get(ax, [xy 'Lim']);
n_tick = numel(S.tick);
for i_tick = 1:n_tick
    tick = S.tick{i_tick};
    if nnz((lim(1) <= tick) & (tick <= lim(2))) >= S.min_n_tick
        break;
    end
end
set(ax, [xy 'Tick'], tick);

