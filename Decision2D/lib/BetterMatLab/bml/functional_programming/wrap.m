function out = wrap(f, out_indices)

% out = wrap(f, out_indices)
% 
% If you've ever needed multiple outputs from a function wrapped up in a
% cell array, you may have felt it was a bit awkward to accomplish without
% doing something like this:
%
% [a, b, c] = f(...)
% x = {a, b, c};
%
% That's not always desirable.
%
% This function makes it much easier. Just pass in a handle for a function
% to execute and numbers indicating which outputs are desired in the cell 
% array.
%
% x = wrap(@() f(...), 1:3)
%
% This is especially useful for accessing multiple outputs from a function
% *inside* an anonymous function, when one can't save numerous outputs as
% in [a, b, c] = f(...).
%
% Tucker McClure
% Copyright 2013 The MathWorks, Inc.

    [out{1:max(out_indices)}] = f();
    
    out = out(out_indices); % debugged by YK
end
