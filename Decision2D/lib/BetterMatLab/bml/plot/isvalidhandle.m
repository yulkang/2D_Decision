function tf = isvalidhandle(h)
% tells if the graphics handle is valid (i.e., not deleted).
%
% tf = isvalidhandle(h)

% 2016 (c) Yul Kang. hk2699 at columbia dot edu.

if isempty(h)
    tf = false;
elseif isscalar(h)
    tf = ishandle(h);
    if ~isnumeric(h) % ~verLessThan('matlab', '8.4')
        tf = tf && isvalid(h);
    end
else
    tf = ishandle(h);
    if ~isnumeric(h(tf)) % ~verLessThan('matlab', '8.4')
        tf(tf) = isvalid(h(tf));
    end
end