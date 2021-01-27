function varargout = strsep(s, sep, ix, toCell)
% STRSEP  Separates string into components and returns varargout
%
% varargout = strsep(s, sep='_', ix, toCell=false)
%
% s   : String or cell array of strings.
% sep : Separaters. sep can have multiple characters. Each character will be considered as a seprator.
% ix  : Which inputs will be returned. ix<=0 will be counted from the last one. 
%       ':' will give all.
% toCell: if true, give all outputs in one cell array.
%
% Superfluous varargout will be assigned ''.
%
% Example:
% >> [aa, bb, cc, dd] = strsep('a_b+c', '_+')
% aa =
% a
% 
% bb =
% b
% 
% cc =
% c
% 
% dd =
%      '' 
%
% Example 2:
% >> [aa, bb] = strsep('a_b+c', '_+', [2 0])
% aa =
% b
% 
% bb =
% c
%
% Example 3:
% [aa, bb, cc, dd] = strsep({'a__bc'; 'ab_c_'})
% aa = 
%     'a'
%     'ab'
% bb = 
%     [1x0 char]
%     'c'       
% cc = 
%     'bc'      
%     [1x0 char]
% dd = 
%     []
%     []
%
% See also fullstr.

if ~exist('sep', 'var'), sep = '_'; end
if nargin < 3
    ix = ':';
end
if nargin < 4
    toCell = false;
end

if ischar(s)
    varargout = separate_str(s, sep);
    
elseif iscell(s)
    n = numel(s);
    
    C = cell(n,1);
    for i_s = 1:n
        c = separate_str(s{i_s}, sep);
        
        C(i_s, 1:length(c)) = c;
    end
    
    varargout = cell(1,size(C,2));
    for i_c = 1:size(C,2)
        varargout{i_c} = reshape(C(:,i_c), size(s));
    end
else
    error('Give only string or cell inputs!');
end

if ~(ischar(ix) && isequal(ix, ':'))
    varargout = varargout(ix);
end
if toCell
    varargout = {varargout};
else
    if length(varargout) < nargout
        if ischar(s)
            varargout((length(varargout)+1):nargout) = {''};
        else % iscell(s)
            varargout((length(varargout)+1):nargout) = {cell(size(s))};
        end
    end
end
end

function c = separate_str(s, sep)
    s = [s, sep(1)];
    sep_loc = find(bsxEq(s(:), sep(:)'));

    c = cell(1, length(sep_loc));

    c_st = 0;
    for i_sep = 1:length(sep_loc)
        c{i_sep} = s((c_st+1):(sep_loc(i_sep)-1));
        c_st = sep_loc(i_sep);
    end
end