function tf = is_valid_variable_name(name)
if ischar(name)
    tf = isequal(regexp(name, '^[a-zA-Z]+\w*$'), 1);
else
    tf = false;
end
