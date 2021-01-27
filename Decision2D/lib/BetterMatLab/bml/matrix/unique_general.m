function [c, ia, ic] = unique_general(a) % , varargin)
% Same as unique() but applies to any class. The order is always stable.
% 
% [v, ia, ic] = unique_general(v0)
%
% EXAMPLE:
% >> [c, ia, ic] = unique_general({2, 1, 4, 1, 'a', 'b', 'a', 4})
% c = 
%     [2]    [1]    [4]    'a'    'b'
% ia =
%      1     2     3     5     6
% ic =
%      1     2     3     2     4     5     4     3% 
%
% >> [c, ia, ic] = unique_general([2 1 2 4 3 4 1])
% c =
%      2     1     4     3
% ia =
%      1     2     4     5
% ic =
%      1     2     1     3     4     3     2
     
% 2016 (c) Yul Kang. hk2699 at columbia dot edu.

if isempty(a)
    c = a;
    ia = [];
    ic = [];
    return;
end

% S = varargin2S(varargin, {
%     'prioritize_last', false
%     });

n = numel(a);
% v = repmat(feval(class(v0)), size(v0));

is_unique = false(size(a));
is_unique(1) = true;
n_unique = 1;
ic = ones(size(a));

for ii = 2:n
    is_unique1 = true;
    
    jjs = hVec(find(is_unique));
%     if S.prioritize_last
%         jjs = flip(jjs, 2);
%     end
    
    for jj = jjs
        if isequal(a(jj), a(ii))
            is_unique1 = false;
            break;
        end
    end
    is_unique(ii) = is_unique1;
    if is_unique1
        n_unique = n_unique + 1;
        ic(ii) = n_unique;
    else
        ic(ii) = ic(jj);
    end
end
ia = find(is_unique);
c = a(is_unique);
% if all(is_unique(:))
%     c = reshape(c, size(a));
% end


% v(1) = v0(1);
% n_unique = 1;
% for ii = 2:n
%     is_unique = true;
%     for jj = 1:n_unique
%         if isequal(v(jj), v0(ii))
%             is_unique = false;
%             break;
%         end
%     end
%     
%     if is_unique
%         n_unique = n_unique + 1;
%         v(n_unique) = v0(ii);
%     end
% end
% 
% v = v(1:n_unique);