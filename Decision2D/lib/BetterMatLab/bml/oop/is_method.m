function [tf, defining_class_name, method_] = ...
    is_method(class_name, method_name, direct_only)
% [tf, defining_class_name, method_] = ...
%     is_method(class_name, method_name, direct_only=false)

if nargin < 3, direct_only = false; end

assert(exist(class_name, 'class') ~= 0, 'Class %s doesn''t exist!', class_name);
mc = meta.class.fromName(class_name);

methods = mc.MethodList;
method_names = {methods.Name};
ix = find(strcmp(method_name, method_names));

if isempty(ix)
    tf = false; 
    return; 
else
    tf = true;
end

method_ = methods(ix);
defining_class_name = method_.DefiningClass.Name;

if direct_only
    tf = strcmp(defining_class_name, class_name);
end