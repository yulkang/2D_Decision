function [tf, res_str] = strcmpStart(starting_str, containing_str)
% [tf, res_str] = strcmpStart(starting_str, containing_str)
%
% tf = strcmpStart(string, string) = strcmpFirst(string, string, 'strict', true)
% tf(k,1) = strcmpStart(cell{k}, string)
% tf(1,k) = strcmpStart(string, cell{k})
% tf(k,m) = strcmpStart(cell{k}, cell{k})
%
% res_str = containing_str(tf)
%
% Example:
% >> strcmpStart({'a', 'ab'}, {'a', 'ab', 'abc', 'bab'})
% ans =
%   2×4 logical array
%    1   1   1   0
%    0   1   1   0
%
% See also: strcmpFirst, strcmpFirsts

% 2015 (c) Yul Kang. yul dot kang dot on at gmail.

if ~iscell(starting_str)
    starting_str = {starting_str};
end
if ~iscell(containing_str)
    containing_str = {containing_str};
end
if isempty(containing_str)
    tf = [];
    res_str = {};
    return;
end

for i_st = numel(starting_str):-1:1
    st = starting_str{i_st};
    
    for i_con = numel(containing_str):-1:1
        con = containing_str{i_con};
        
        tf(i_st, i_con) = strcmpFirst(st, con, 'strict', true);
    end
end

if nargout >= 2
    res_str = containing_str(tf); 
end