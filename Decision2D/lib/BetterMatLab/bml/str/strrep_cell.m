function c = strrep_cell(c, s1, s2, varargin)
% Replace entries matching s1{k} with s2{k} (or s{k,1} with s{k,2}).
%
% c = strrep_cell(c, s1, s2);
% c = strrep_cell(c, {
%       s1_1, s1_2
%       s1_2, s2_2
%       ...
%       });
% c = strrep_cell(c, s1, [], 'wholeStringOnly', false)
%
% See also: STRREP, STRREPS, PSYLIB

S = varargin2S(varargin, {
    'wholeStringOnly', false
    });

% Two-column cell array input
if nargin < 3 || isempty(s2)
    if isempty(s1)
        % Nothing to change
        return;
    end
    s2 = s1(:,2);
    s1 = s1(:,1);
end

% Match length
n = max(length(s1), length(s2));
if ischar(s1),   s1 = {s1}; end
if isscalar(s1), s1 = repmat(s1, [1, n]); end
if ischar(s2),   s2 = {s2}; end
if isscalar(s2), s2 = repmat(s2, [1, n]); end

% Run strrep
for ii = 1:n
    if iscell(c)
        c(cellfun(@isempty, c)) = {''};
    end
    
    if ~S.wholeStringOnly
        % Find substring, too.
        c = strrep(c, s1{ii}, s2{ii});
    else
        % Only find the whole string.
        tf = strcmp(s1{ii}, c); 

        if any(tf)
            if iscell(c)
                c{tf} = s2{ii};
            else
                c = s2{ii};
            end
        end
    end
end