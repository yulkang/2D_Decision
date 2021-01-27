function [C2, C1] = varargin2C2(C, C0)
% [C2, C1] = varargin2C2(C0)
%
% C: cell array, e.g., varargin, or struct, as from varargin2S.
% C0: default cell array of struct.
%
% C, C0, C1, and C2 can be any of the following:
% - C1-format: {name1, value1, name2, value2, ...}
% - C2-format: {name1, value1; name2, value2; ...}
% - struct
%
% In C and C0, 
% - name can be either a string or a cell array of strings.
% - name as a cell array of strings can be useful for factorizeC.
% - names may not be unique. Priority is given to:
%   the last one in C > the last one in C0.
% - The order of the rows in the result follows that of C0,
%   with rows with new names being added at the end, and
%   with rows with old names being updated by latter rows in C0 then in C1.
%
% C1: 1 x (n*2) cell array of name-value pairs.
% C2: n x 2 cell array of name-value pairs.
%
% See also: factorizeC, varargin2C, varargin2S
%
% EXAMPLE:
% [C2, C1] = bml.args.varargin2C2({
%     'a', 1
%     'b', 2
%     {'a', 'b'}, {3,4}
%     'a', 10
%     {'c', 'd'}, {5,6}
%     {'a', 'b'}, {30, 40}
%     }, {
%     'a', 11
%     'g', 22
%     {'e', 'f'}, {33,44}
%     'a', 101
%     {'c', 'd'}, {51,61}
%     {'e', 'f'}, {301, 401}
%     })
% 
% C2_expected = {
%     'a', 10
%     'g', 22
%     {'e', 'f'}, {301, 401}
%     {'c', 'd'}, {5, 6}
%     'b', 2
%     {'a', 'b'}, {30, 40}
%     };
% C1_expected = bml.args.C2_to_C(C2_expected);
% 
% assert(isequal(C2, C2_expected));
% assert(isequal(C1, C1_expected));
    
% 2016 (c) Yul Kang. hk2699 at columbia dot edu.

if nargin < 2
    C0 = {};
end

C2_new = S2C2(C);
C2_def = S2C2(C0);

C2 = cell(0,2);
C2 = apply_C2(C2_def, C2);
C2 = apply_C2(C2_new, C2);

C1 = bml.args.C2_to_C(C2);
end

%%
function C_res = apply_C2(C2, C20)
    % flip to prioritize latter entries.
    
    C_all = [C20; C2];
    
    names = unique_general(C_all(:,1));
    n_unique = length(names);
    C_res = cell(n_unique, 2);
    C_res(:,1) = names;
    
    for ii = 1:n_unique
        ix = find(cellfun(@(v) ...
                isequal(names{ii}, v), ...
                C_all(:,1)), ...
            1, 'last');
        C_res(ii,2) = C_all(ix,2);
    end
end

%%
function test
    %%
    [C2, C1] = bml.args.varargin2C2({
        'a', 1
        'b', 2
        {'a', 'b'}, {3,4}
        'a', 10
        {'c', 'd'}, {5,6}
        {'a', 'b'}, {30, 40}
        })
    celldisp(C2(3:4,:));
    
    %%
    [C2, C1] = bml.args.varargin2C2({
        'a', 1
        'b', 2
        {'a', 'b'}, {3,4}
        'a', 10
        {'c', 'd'}, {5,6}
        {'a', 'b'}, {30, 40}
        }, {
        'a', 11
        'g', 22
        {'e', 'f'}, {33,44}
        'a', 101
        {'c', 'd'}, {51,61}
        {'e', 'f'}, {301, 401}
        })
    
    C2_expected = {
        'a', 10 % the first name in the second input, overwritten by a latter row.
        'g', 22 % the second name in the second input, never overwritten.
        {'e', 'f'}, {301, 401} % the third name in the second input, overwritten by a latter row.
        {'c', 'd'}, {5, 6} % appeared only in the first input
        'b', 2
        {'a', 'b'}, {30, 40}
        };
    C1_expected = bml.args.C2_to_C(C2_expected);
    
    assert(isequal(C2, C2_expected));
    assert(isequal(C1, C1_expected));
end