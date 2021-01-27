function c = strOrder(c, f1, op, f2)
% c = strOrder(c, f1, op, f2)
%
% c: cell array of strings.
% op: 'bef', 'aft', 'first', or 'last'.
% f1: a string or a cell array of strings to move position.
% f2: (for 'bef' and 'aft' only) string at a reference position.
%
% EXAMPLE
% -------
% strOrder({'d', 'a', 'f', 'b'}, 'c', 'bef', 'b')
% ans = 
%     'd'    'a'    'f'    'c'    'b'
% 
% strOrder({'d', 'a', 'f', 'b'}, 'c', 'aft', 'b')
% ans = 
%     'd'    'a'    'f'    'b'    'c'
% 
% strOrder({'d', 'a', 'f', 'b'}, 'c', 'aft', 'd')
% ans = 
%     'd'    'c'    'a'    'f'    'b'
% 
% strOrder({'d', 'a', 'f', 'b'}, 'c', 'bef', 'd')
% ans = 
%     'c'    'd'    'a'    'f'    'b'
% 
% strOrder({'d', 'a', 'f', 'b'}, {'ccc', 'c', 'cc'}, 'bef', 'd')
% ans = 
%     'ccc'    'c'    'cc'    'd'    'a'    'f'    'b'
% 
% strOrder({'d', 'a', 'f', 'b'}, {'ccc', 'c', 'cc'}, 'bef', 'b')
% ans = 
%     'd'    'a'    'f'    'ccc'    'c'    'cc'    'b'
% 
% strOrder({'d', 'a', 'f', 'b'}, {'ccc', 'c', 'cc'}, 'aft', 'b')
% ans = 
%     'd'    'a'    'f'    'b'    'ccc'    'c'    'cc'
% 
% strOrder({'d', 'a', 'f', 'b'}, {'ccc', 'c', 'cc'}, 'aft', 'd')
% ans = 
%     'd'    'ccc'    'c'    'cc'    'a'    'f'    'b'
%
% 2015 (c) Yul Kang. hk2699 at cumc dot columbia dot edu.

if nargin < 4, f2 = ''; end

siz = size(c);
c = c(:)'; % TODO - may optimize

if iscell(f1)
    if length(f1) == 1
        f1 = f1{1};
    else
        if nargin < 4, f2 = ''; end
        c = strOrder(c, f1{1}, op, f2);
        f1_ix = find(strcmp(f1{1}, c));
        c = [c(1:f1_ix), hVec(f1(2:end)), c((f1_ix+1):end)];
        
        c = reshape(c, siz);
        return;
    end
end

switch op
    case 'bef'
        c = setdiff(c, f1, 'stable'); % TODO - may optimize
        f2_ix = find(strcmp(f2, c));
        
        c = [c(1:(f2_ix-1)), {f1}, c(f2_ix:end)]; % TODO - may optimize
        
    case 'aft'
        c = setdiff(c, f1, 'stable');
        f2_ix = find(strcmp(f2, c));
        
        c = [c(1:f2_ix), {f1}, c((f2_ix+1):end)];
        
    case 'first'
        c = setdiff(c, f1, 'stable');
        c = [{f1}, c];
        
    case 'last'
        c = setdiff(c, f1, 'stable');
        c = [c, {f1}];
end

c = reshape(c, siz);
end