function tf = isColorSpec(v)
tf = isnumeric(v) || ...
    (ischar(v) && length(v)==1 && any(v == 'wkrgbmcy'));
end