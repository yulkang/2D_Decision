function [success, msg, msgid] = mkdir2(d)
% Skips making directory if already exists.
%
% [success, msg, msgid] = mkdir2(d)

if iscell(d)
    [success, msg, msgid] = cellfun(@bml.file.mkdir2, d, 'UniformOutput', false);
elseif bml.file.exist_dir(d)
    success = false;
    msg = '';
    msgid = '';
else
    [success, msg, msgid] = mkdir(d); 
end