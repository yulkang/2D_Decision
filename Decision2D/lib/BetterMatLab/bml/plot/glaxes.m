function [hgl_out, varargout] = glaxes(ha, op, varargin)
% Global title and labels for an array of axes.
%
% hs:   an array of axes handles.
% op:   'title', 'xlabel', 'ylabel', or 'set'
% ht:   Handle of the text object.
% hgl:  Handle of the global axis object.
%
% 2014 (c) Yul Kang. hk2699 at columbia dot edu.

%% Find one enclosing the given axes
fig = get(ha(1), 'Parent');
u   = get(fig, 'UserData');

if ~isfield(u, 'glaxes')
    u.glaxes = [];
end

hgl = [];
nu  = length(u.glaxes);
to_remove = false(1, nu);
for ii = 1:nu
    if isequal(u.glaxes(ii).ha, ha)
        if isvalidhandle(u.glaxes(ii).hgl)
            hgl = u.glaxes(ii).hgl;
            break;
        else % e.g., when hgl is deleted due to clf.
            to_remove(ii) = true;
        end
    end
end

%% Delete invalid entries
u.glaxes(to_remove) = [];
nu = length(u.glaxes);

%% Make one if not found
if isempty(hgl)
    hgl = axes;
    u.glaxes(nu+1).ha   = ha;
    u.glaxes(nu+1).hgl  = hgl;
end
set(fig, 'UserData', u);

%% By default, axis off and put at the bottom.
axis(hgl, 'off');
uistack(hgl, 'bottom');

%% Set position to enclose all axes
pos = get(ha, 'Position');
if iscell(pos), pos = cell2mat(pos); end

posSW = min(pos(:,1:2), [], 1);
posNE = max(pos(:,1:2) + pos(:,3:4), [], 1);
set(hgl, 'Position', [posSW, posNE - posSW]);

%% Perform op
if nargin >= 2
    switch op
        case {'xlabel', 'ylabel'}
            set(get(hgl, op), 'Visible', 'on');
    end

    [varargout{1:(nargout-1)}] = feval(op, hgl, varargin{:});        
end

%% Output
if nargout > 0, hgl_out = hgl; end