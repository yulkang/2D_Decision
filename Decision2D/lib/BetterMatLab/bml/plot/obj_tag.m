function h = obj_tag(typ, tag, varargin)
% h = obj_tag(typ, tag, varargin)
%
% See also fig_tag, axes_tag
%
% 2014 (c) Yul Kang. hk2699 at columbia dot edu.

tag = safe_name(tag);

S_props0 = varargin2S(varargin);
S_props1 = struct;
if strcmpi(typ, 'figure')
    if isfield(S_props0, 'Position')
%         S_props0 = varargin2S({
%             'WindowStyle', 'normal'
%             }, S_props0);
%         S_props1 = varargin2S({
%             'Position', S_props0.Position
%             });
        S_props0 = rmfield(S_props0, 'Position');
    end
end
C_props0 = varargin2C({'Tag', tag}, S_props0);
C_props1 = varargin2C({'Tag', tag}, S_props1);

h = findobj('Tag', tag, 'Type', typ);
if isempty(h)
    h = findobj('Name', tag, 'Type', typ);
end

if ~isempty(h)
    try
        % Focus the obj
        switch typ
            case 'figure'
                figure(h);
            case 'axes'
                axes(h);
            otherwise
                feval(typ, h);
        end
        
        if ~isempty(C_props0)
            % Set obj properties
            set(h, C_props0{:});
        end
        
        % If succeed, skip creating a new obj.
        return;
    catch err
        % When the obj is deleted or invalid, just issue warning,
        % and create a new obj in the line below.
        warning(err_msg(err));
    end
end

% Create a new obj with the specified tag and properties
if is_in_parallel
    C_props0 = varargin2C({
        'WindowStyle', 'normal'
        'Visible', 'off'
        'Tag', tag
        }, C_props0);
    prev_win_style = get(0, 'DefaultFigureWindowStyle');
    set(0, 'DefaultFigureWindowStyle', 'normal');
end
switch lower(typ)
    case 'figure'
        h = figure(C_props0{:});
    case 'axes'
        h = axes(C_props0{:});
    otherwise
        h = feval(typ, C_props0{:});
end
if is_in_parallel
    set(0, 'DefaultFigureWindowStyle', prev_win_style);
end

% Set the position after setting WindowStyle
if ~isempty(C_props1)
    set(h, C_props1{:});
end
