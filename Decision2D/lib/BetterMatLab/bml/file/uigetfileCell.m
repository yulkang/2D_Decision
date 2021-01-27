function res = uigetfileCell(varargin)
% UIGETFILECELL   returns cell array of file names through UIGETFILE.
%
% res = UIGETFILECELL(filt, opt)
%
% filt  : A string filter, like 'dirname/*.ext' (which evokes uigetfile), or
%         an existing file's name, like 'dirname/filename.ext', or
%         a cell array of existing files' name, like 
%         {'dirname/filename1.ext', 'dirname/filename2.ext'}.
%
% opt   : see help UIGETFILE.
%
% res   : Always a cell array, e.g.,
%         {'dirname/filename1.ext', 'dirname/filename2.ext'}
%         In case a filter is given, the file names are sorted in ascending order.
%
% See also UIGETFILE, DIRCELL.

if nargin == 0, varargin{1} = ''; end

if ischar(varargin{1}) && exist(varargin{1}, 'file')
    res = varargin(1);
    
elseif iscell(varargin{1}) && ...
       all(cellfun(@(fname) exist(fname, 'file'), varargin{1}))
    
    res = varargin{1};
    
else
    if numel(varargin) >= 1
        varargin = [varargin(1), varargin2C(varargin(2:end), {
            'MultiSelect', 'on'
            })];
    end
    [r, pth] = uigetfile(varargin{:});

    if isequal(r, 0)
        res = {};

    elseif ischar(r)
        res = {fullfile(pth, r)};

    elseif iscell(r)
        res = cellfun(@(f) fullfile(pth,f), sort(r), 'UniformOutput', false);

    else
        error('Unparseable result from uigetfile!');
    end
    
    
end