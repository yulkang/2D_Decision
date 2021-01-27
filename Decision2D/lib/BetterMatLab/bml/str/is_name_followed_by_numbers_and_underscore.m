function [tf, names] = is_name_followed_by_numbers_and_underscore(name, names)
% [tf, names] = is_name_followed_by_numbers_and_underscore(name, names)
%
% EXAMPLE:
% >> tf = is_name_followed_by_numbers_and_underscore('abc', {
%     'ab', 'abc', 'abc1_', 'abc1_b', 'abc1_23', 'abc1_b1_23'})
% tf =
%      0     0     1     0     1     0
%
% 2015 (c) Yul Kang. hk2699 at cumc dot columbia dot edu.

ix = regexp(names, ['^' name '[0-9_]+$']);
tf = ~cellfun(@isempty, ix);

if nargout >= 2
    names = names(tf);
end
end