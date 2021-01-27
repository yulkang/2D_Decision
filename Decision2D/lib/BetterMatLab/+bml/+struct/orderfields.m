function S = orderfields(S, fs, kind, varargin)
% S = orderfields(S, fs, kind)
%
% EXAMPLE:
% >> S = varargin2S({'a', 1, 'b', 2, 'c', 3})
% S = 
%     a: 1
%     b: 2
%     c: 3
%
% >> S = bml.struct.orderfields(S, {'a', 'b'}, 'last')
% S = 
%     c: 3
%     a: 1
%     b: 2
% 
% >> S = bml.struct.orderfields(S, {'b', 'a'}, 'last')
% S = 
%     c: 3
%     b: 2
%     a: 1
% 
% >> S = bml.struct.orderfields(S, {'b', 'a'}, 'first')
% S = 
%     b: 2
%     a: 1
%     c: 3
% 
% >> S = bml.struct.orderfields(S, {'c'}, 'first')
% S = 
%     c: 3
%     b: 2
%     a: 1
%
% 2016 (c) Yul Kang. hk2699 at columbia dot edu.
    
opt = varargin2S(varargin, {
    'ignore_absent_field', true
    });

if ischar(fs), fs = {fs}; end
assert(iscell(fs));
assert(all(cellfun(@ischar, fs)));
fs = fs(:);

if ~exist('kind', 'var')
    kind = 'perm';
end

fs0 = fieldnames(S);
switch kind
    case 'perm'
        fs1 = fs;
    case 'first'
        incl = ismember(fs0, fs);
        fs1 = [fs; fs0(~incl)];
    case 'last'
        incl = ismember(fs0, fs);
        fs1 = [fs0(~incl); fs];
end

if opt.ignore_absent_field
    % Otherwise, if fs1 contains fields absent in fs0, will evoke error.
    fs1 = fs1(ismember(fs1, fs0));
end

S = orderfields(S, fs1);