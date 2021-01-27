function dst = add_subdir_ext(src, subdir, ext)
% dst = add_subdir_ext(src, subdir='', ext='')
%
% Exsiting extension, if any, is replaced with ext.
%
% 2016 (c) Yul Kang. hk2699 at columbia dot edu.

if ~exist('subdir', 'var'), subdir = ''; end
if ~exist('ext', 'var'), ext = ''; end

[pth, nam] = fileparts(src);
dst = fullfile(pth, subdir, [nam, ext]);