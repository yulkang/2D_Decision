function [ax, h_children, h_gap] = break_axis(ax0, xy_to_break, lim1, lim2, varargin)
% Breaks an axes AX into two axes with lim1 and lim2 as their x or ylim.
%
% [ax, h_children, h_gap] = BREAK_AXIS(ax0, xy, lim1, lim2, ...)
%
% lim1, lim2: each a two-vector or a scalar.
% ax(1) : the left axes if broken abscissa, bottom if broken ordinate.
% ax(2) : the remaining half.
% ax(3) : a third axes containing a patch to mark the break.
% ax(4) : a fourth axes containing the label of the broken axes.
% hp : handle of the patch that marks the break.
%
% The function works by:
% - Adding two new axes, and
% - Keeping the xlabel or ylabel of the broken axes, if any.
% - Adding a patch to a third axes above the 
%
% NOTE
% - After using this function, subplot will no longer work as expected,
%   so use after the original axes is in intended position.
% - Legends are not kept for now.


% 2016 (c) Yul Kang. hk2699 at columbia dot edu.

%% EXAMPLE
if nargin == 0
    clf;
    ax0 = axes;
    plot([1 2 3 1.2], [1 3 2 1]);
    box(ax0, 'off');
    set(ax0, 'TickDir', 'out');
    xy_to_break = 'x'; % try both x and y
    lim1 = 1.45;
    lim2 = 1.9;
    title(sprintf('TITLE\n'));
    xlabel('XLABEL');
    ylabel('YLABEL');
    text(2.5, 2, 'TEXT');
    
    % legends are not kept for now. 
    legend('LEGEND'); 
end

%% Options
S = varargin2S(varargin, {
    ... % Margin as a proportion of the original length of the broken axis.
    'margin', 0 
    ...
    ... % Location of the center of the gap
    ... % as a proportion of the original length of the broken axis.
    'prop', 0.5 
    ...
    ... % The gap should occlude the split axes beyond the margin
    ... % to occlude the kept axis of the right/top plot.
    ... % Specify in the unit of the proportion of the kept/broken axis.
    'extend_gap_along_kept_axis', [0.01, 0.01]
    'extend_gap_along_broken_axis', [0, 0.02]
    });

%% Interpret inputs
assert(strcmp(get(ax0, 'Type'), 'axes'));

xy_to_break = upper(xy_to_break);
assert(ismember(xy_to_break, 'XY'));
xy_to_keep = setdiff('XY', xy_to_break);

brlim = [xy_to_break, 'Lim'];
kplim = [xy_to_keep, 'Lim'];
lim0 = get(ax0, brlim);

assert(isnumeric(lim1));
if isscalar(lim1)
    lim{1} = [lim0(1), lim1];
else
    assert(numel(lim1) == 2);
    lim{1} = lim1;
end

assert(isnumeric(lim2));
if isscalar(lim2)
    lim{2} = [lim2, lim0(2)];
else
    assert(numel(lim2) == 2);
    lim{2} = lim2;
end

%% Position new axes and set limits on the broken axis.
% Get position
set(ax0, 'Units', 'normalized');
pos0 = get(ax0, 'Position');

% Fix kept axis
set(ax0, kplim, get(ax0, kplim));

% Copy axes
fig = get(ax0, 'Parent');
ax(1) = copyobj(ax0, fig);
ax(2) = copyobj(ax0, fig);

% Set new positions
pos{1} = pos0;
pos{2} = pos0;

switch xy_to_break
    case 'X'
        brloc = 1; % broken axis's location = left
        brsiz = 3; % broken axis's size = width
        kploc = 2; % kept axis's location = bottom
        kpsiz = 4; % kept axis's size = height
        
    case 'Y'
        brloc = 2; % bottom
        brsiz = 4; % height
        kploc = 1; % left
        kpsiz = 3; % width
end
pos{1}(brsiz) = pos0(brsiz) * S.prop - S.margin / 2;
pos{2}(brsiz) = pos0(brsiz) * (1 - S.prop) - S.margin / 2;
pos{2}(brloc) = pos0(brloc) + pos0(brsiz) * S.prop + S.margin / 2;

set(ax(1), 'Position', pos{1}, brlim, lim{1});
set(ax(2), 'Position', pos{2}, brlim, lim{2});

%% The gap is drawn on a third axes.
ax(3) = copyobj(ax0, fig);

pos{3} = pos0;
set(ax(3), 'Color', 'none');
axis(ax(3), 'off'); % DEBUG

brext = S.extend_gap_along_broken_axis;
kpext = S.extend_gap_along_kept_axis;

pos{3}(kploc) = pos0(kploc) - pos0(kpsiz) * kpext(1);
pos{3}(kpsiz) = pos0(kpsiz) * (1 + sum(kpext));
set(ax(3), 'Position', pos{3});

brlim0 = get(ax0, brlim);
brlimsiz = diff(brlim0);
f_prop2br = @(prop) prop .* brlimsiz + brlim0(1);

kplim0 = get(ax0, kplim);
kplimsiz = diff(kplim0);
f_prop2kp = @(prop) prop .* kplimsiz + kplim0(1);

brlim3 = brlim0;
kplim3 = f_prop2kp([0 - S.extend_gap_along_kept_axis(1), ...
                   1 + S.extend_gap_along_kept_axis(2)]);

%%
set(ax(3), ...
    brlim, brlim3, ...
    kplim, kplim3);
               
delete(get(ax(3), 'Children'));

pos_gap(brloc) = f_prop2br(S.prop - (S.margin + sum(brext)) / 2);
pos_gap(brsiz) = (S.margin + sum(brext)) .* brlimsiz;
pos_gap(kploc) = kplim3(1);
pos_gap(kpsiz) = diff(kplim3);

%%
h_gap.patch = patch(ax(3), ...
    pos_gap(1) + [0 1 1 0] * pos_gap(3), ...
    pos_gap(2) + [0 0 1 1] * pos_gap(4), ...
    'w', ...
    'EdgeColor', 'none');

%%
switch xy_to_break
    case 'X'
        h_gap.line(1) = line(ax(3), ...
            pos_gap(brloc) + [0 0], ...
            kplim3, ...
            'Color', 'k');
        hold on;
        h_gap.line(2) = line(ax(3), ...
            pos_gap(brloc) + pos_gap(brsiz) + [0 0], ...
            kplim3, ...
            'Color', 'k');
        hold off;
        
    case 'Y'
        h_gap.line(1) = line(ax(3), ...
            kplim3, ...
            pos_gap(brloc) + [0 0], ...
            'Color', 'k');
        hold on;
        h_gap.line(2) = line(ax(3), ...
            kplim3, ...
            pos_gap(brloc) + pos_gap(brsiz) + [0 0], ...
            'Color', 'k');
        hold off;
end

%% Copy title and axis label onto the first axes.
brlabel = [xy_to_break, 'Label'];
lab0 = get(ax0, brlabel);
title0 = get(ax0, 'Title');

% Remove unnecessary labels
for ii = 1:3
    ax1 = ax(ii);
    lab1 = get(ax1, brlabel);
    set(lab1, 'String', '');
    title1 = get(ax1, 'Title');
    set(title1, 'String', '');
end

% limits can change so we need to use normalized.
% set(lab0, 'Units', 'normalized'); 
poslab0 = get(lab0, 'Position');
% 
% set(title0, 'Units', 'normalized');
% postitle0 = get(title0, 'Position');
% 
% poslab1 = poslab0;
% poslab1(brloc) = poslab0(brloc) * pos0(brsiz) / pos{1}(brsiz);

% postitle1 = postitle0;
% postitle1(brloc) = postitle0(brloc) * pos0(brsiz) / pos{1}(brsiz);

lab1 = copyobj(lab0, ax(3));
% lab1 = get(ax(1), brlabel);
% set(lab1, 'Units', 'normalized');
% set(lab1, 'Position', poslab1);
set(lab1, 'Position', poslab0);

title1 = copyobj(title0, ax(3));
% title1 = get(ax(1), 'Title');
% set(title1, 'Units', 'normalized');
% set(title1, 'Position', postitle1);

h_children.label = lab1;
h_children.title = title1;

%% Remove unnecessary parts of ax(2)
set(ax(2), [xy_to_keep, 'Tick'], []);
kplab = get(ax(2), [xy_to_keep, 'Label']);
set(kplab, 'String', '');

%% Stack ax(3) on ax(1) on ax(2) 
% so that the title and the axis label comes on top, 
% the gap comes next, and the unnecessary parts of the ax(2) is hidden.
uistack(ax(1), 'top');
uistack(ax(3), 'top');

%% Delete the original axes.
delete(ax0);