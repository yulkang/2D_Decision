function varargout = help(obj, attr)
% help(obj, [meth])
% help('class', [meth])
% help(file)
% [...] = help(...)

% 2015-2016 (c) Yul Kang. hk2699 at columbia dot edu.

if exist('attr', 'var') && ~isempty(attr)
    if isobject(obj)
        cl = class(obj);
%         info = metaclass(obj);
    else
        assert(ischar(obj));
        assert(exist(obj, 'class'));
        cl = obj;
%         info = ?cl;
    end

    % Unlike edit, help may exist for properties.    
%     [is_prop, prop_info] = bml.oop.is_prop(info, attr);
%     if is_prop
%         help(prop_info.DefiningClass.Name);
%     else
        help([cl '.' attr]);
%     end
    return;
    
elseif isobject(obj)
    [varargout{1:nargout}] = help(class(obj));
    return;
    
elseif ischar(obj)
    try
        [varargout{1:nargout}] = help(obj);
        return;
    catch
        obj = evalin('caller', ['@' obj]);
    end
end
    
if isa(obj, 'function_handle')
    obj = func2str(obj);
    obj = strrep(obj, '@(varargin)', '');
    obj = strrep(obj, '(varargin{:})', '');
    ix = find(obj == '.', 1, 'last');
    obj0 = obj(1:(ix - 1));
    attr = obj((ix + 1):end);
    cl = evalin('caller', ['class(' obj0 ')']);
    [varargout{1:nargout}] = help([cl '.' attr]);
    
else
    error('Unknown input class: %s\n', class(obj));
end
