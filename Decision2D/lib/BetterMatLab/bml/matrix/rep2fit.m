function v = rep2fit(v, sizRes, varargin)
% res = rep2fit(src, sizRes, ...)
% 
% Applies repmat to src so that size(res) == sizRes.
% If assert_multiple = false (default),
% if sizRes is not a multiple of size(src), truncates the leftover (slow).
% If assert_multiple = true, throws an error if sizRes is not a multiple of
% size(src).
%
% OPTIONS:
% 'assert_multiple', false
%
% Ex:
%
% >> rep2fit(magic(3), [2 5])
% ans =
%      8     1     6     8     1
%      3     5     7     3     5
%
% >> rep2fit(magic(3), [2 5], 'assert_multiple', true)
% sizSrc: 3 3
% sizRes: 2 5
% Error using rep2fit (line 39)
% sizRes is not a multiple of sizRes!
% 
% >> rep2fit(magic(3), [3 6], 'assert_multiple', true)
% ans =
%      8     1     6     8     1     6
%      3     5     7     3     5     7
%      4     9     2     4     9     2
%
% See also: rep2match.
%
% 2013 (c) Yul Kang, hk2699 at columbia dot edu.

if isequal(size(v), sizRes), return; end

S = varargin2S(varargin, {
    'assert_multiple', false
    });

sizSrc = ones(1, length(sizRes));
sizSrc(1:ndims(v)) = size(v);

v = repmat(v, ceil(sizRes ./ sizSrc));

if any(mod(sizRes, sizSrc))
    if S.assert_multiple
        fprintf('sizSrc:'); fprintf(' %d', sizSrc); fprintf('\n');
        fprintf('sizRes:'); fprintf(' %d', sizRes); fprintf('\n');
        error('sizRes is not a multiple of sizRes!');
    end
    
    s = struct('type', {'()'}, 'subs', {{}});
    
    for ii = length(sizRes):-1:1
        s.subs{ii} = 1:sizRes(ii);
    end
    
    v = subsref(v, s);
end    
end