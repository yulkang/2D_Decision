function arg = arg2C(arg)
% Convert struct or N x 2 cell array to name-value pairs.
%
% EXAMPLE:
% >> arg2C({'a', 1; 'b', 2}) % N x 2 cell arrays are converted to name-value pairs.
% ans = 
%     'a'    [1]    'b'    [2]
% 
% >> arg2C(struct('a', 1, 'b', 2)) % Structs are converted to name-value pairs.
% ans = 
%     'a'    [1]    'b'    [2]
%     
% >> arg2C({'a', 1, 'b', 2}) % Name-value pair inputs are unchanged.
% ans = 
%     'a'    [1]    'b'    [2]
    
if isstruct(arg)
    arg = S2C(arg);
    
elseif ismatrix(arg)
    arg = arg';
    arg = arg(:)';
end