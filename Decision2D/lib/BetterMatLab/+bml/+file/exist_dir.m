function tf = exist_dir(d)
% Tests for existence of a directory. More accurate than exist(d, 'dir'),
% since the latter returns nonzero even when d is somewhere else in path,
% not in the local relative directory.
%
% 2016 (c) Yul Kang. hk2699 at columbia dot edu.

if ~exist(d, 'dir')
    tf = false;
else
    info = dir(d);
    if isempty(info)
        tf = false;

    else
        names = {info.name};
        is_dot = strcmp('.', names);
        is_dir = [info.isdir];
        tf = any(is_dot & is_dir);
    end
end