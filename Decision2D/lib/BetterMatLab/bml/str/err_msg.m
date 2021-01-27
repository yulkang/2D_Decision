function msg_out = err_msg(err)
% err_msg  Gives error message with links without rethrowing it.
%
% err_msg(err)
% : Displays error message with links.
%   err is an error structure caught by try..catch statements.
%
% msg = err_msg(err)
% : Returns error string with links without displaying it.
%   You can display it by disp(msg).
%
% EXAMPLE:
% >> try arrayfun(@(v) min(v, 2, []), 1:3); catch err, err_msg(err); end
% MIN with two matrices to compare and a working dimension is not supported.
% > In @(v)min(v,2,[]) at 0
%
% See also cmd2link
%
% 2013 (c) Yul Kang. hk2699 at columbia dot edu

msg = err.message;

for i_stack = 1:length(err.stack)
    msg = sprintf('%s\n> In <a href="%s">%s at %d</a>', ...
        msg, ...
        stack2link(err.stack(i_stack)), ...
        err.stack(i_stack).name, ...
        err.stack(i_stack).line);
end

if nargout == 0
    disp(msg);
else
    msg_out = msg;
end
end

function l = stack2link(s)
    l = sprintf('matlab: opentoline(''%s'',%d,1)', ...
        s.file, s.line);
end