function tf = strfinds(str, pattern, varargin)
% STRFINDS  Return indices of text(s) that include the pattern(s).
%
% tf = strfinds(cellstr, pattern)
% : Unlike strfind, strfinds returns a logical array, not a cell array.
%   tf(i_str, 1) = true if pattern appears in cellstr{i_str}.
% 
% tf = strfinds(cellstr, cell_pattern)
% : tf(i_str, i_ptn) = true if cell_pattern{i_ptn} appears 
%   in cellstr{i_str}.
% 
% ix = strfinds(cellstr, pattern, 'numeric')
% : Returns numeric indices.
%
% ix = strfinds(cellstr, pattern, 'first')
% ix = strfinds(cellstr, pattern, 'last')
% : First/last index.
%
% tf = strfinds(cellstr, pattern, 'from')
% tf = strfinds(cellstr, pattern, 'to')
% : True from the first/to the last index.
%
% See also STRFIND

if iscell(str) && ischar(pattern)
    tf = ~cellfun(@isempty, strfind(str, pattern));
elseif ischar(str) && iscell(pattern)
    tf = cellfun(@(p) ~isempty(strfind(str, p)), pattern);
elseif iscell(str) && iscell(pattern)
    assert(isvector(str));
    assert(isvector(pattern));
    
    n_str = numel(str);
    n_ptn = numel(pattern);
    
    tf = false(n_str, n_ptn);
    
    for i_ptn = n_ptn:-1:1
        tf(:, i_ptn) = strfinds(str, pattern{i_ptn}, varargin{:});
    end
else
    error('STR and PATTERN should be a string or a cell array of strings!');
end

if nargin >= 3
    opt = varargin{1};
else
    opt = 'logical';
end
    
switch opt
    case 'logical'
        % do nothing.
    case 'numeric'
        tf = find(tf);
    case 'first'
        tf = find(tf, 1, 'first');
    case 'last'
        tf = find(tf, 1, 'last');
    case 'from'
        ix = find(tf, 1, 'first');
        tf(1:(ix-1)) = false;
        tf(ix:end) = true;
    case 'to'
        ix = find(tf, 1, 'last');
        tf(1:ix) = true;
        tf((ix+1):end) = false;
end
