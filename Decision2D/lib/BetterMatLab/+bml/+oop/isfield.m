function tf = isfield(obj, f)
% Generalized isfield() for objects, datasets, tables, and structs.
%
% 2016 (c) Yul Kang. hk2699 at columbia dot edu.

if isstruct(obj)
    tf = isfield(obj, f);
elseif istable(obj)
    tf = ismember(f, obj.Properties.VariableNames);
elseif isa(obj, 'dataset')
    tf = isdscolumn(obj, f);
elseif isobject(obj)
    tf = isprop(obj, f);
else
    error('ISFIELD:UNSUPPORTEDCLASS', 'Unsupported class %s!', obj);
end