% initialize paths

% 2021 Yul Kang. hk2699 at caa dot columbia dot edu.

%%
addpath(genpath(fullfile('../lib')));
pth0 = genpath('.');
pths = strsep(pth0, pathsep, ':', true);
ix = any(strcmpStart({fullfile('.', 'UNUSED')}, pths), 1);
pths = pths(~ix);
pth = [pathsep, str_bridge(pathsep, pths)];
addpath(pth);

% % function init_path
% restoredefaultpath;
% addpath(genpath('lib'));

% if ~verLessThan('matlab', '9.0')
%     import bml.file.edit
%     import bml.file.help
% end

import bml.override.*

dbstop if error

try
    opengl hardwarebasic
catch err
    warning(err_msg(err));
end

set(0, 'DefaultFigureWindowStyle', 'docked');
set(0, 'DefaultFigureColor', 'w');
varargin = {};
nargin = 0;

global to_use_parallel
to_use_parallel = true;