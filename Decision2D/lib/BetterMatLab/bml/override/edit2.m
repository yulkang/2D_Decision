function edit2(obj, attr)
% edit2(obj, [meth])
% edit2('class', [meth])
% edit2(file)
%
% 2015-2016 (c) Yul Kang. hk2699 at columbia dot edu.

if nargin == 0
    edit;
end

if exist('attr', 'var') && ~isempty(attr)
    if isobject(obj)
        cl = class(obj);
        info = metaclass(obj);
    else
        assert(ischar(obj));
        assert(exist(obj, 'class'));
        cl = obj;
        info = ?cl;
    end
    
    [is_prop, prop_info] = bml.oop.is_prop(info, attr);
    if is_prop
        edit(prop_info.DefiningClass.Name);
    else
        edit([cl '.' attr]);
    end
    return;
    
elseif isobject(obj)
    edit(class(obj));
    return;
    
elseif ischar(obj)
    try
        edit(obj);
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
    edit([cl '.' attr]);
    
else
    error('Unknown input class: %s\n', class(obj));
end
