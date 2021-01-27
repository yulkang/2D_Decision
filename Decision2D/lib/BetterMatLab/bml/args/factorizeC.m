function [Ss, n] = factorizeC(C, fields, to_fix)
% C: struct or cell array with name-value pairs.
% When C is a struct, identical to factorizeS.
% When C is a N-by-2 cell array with C(:,1) names and C(:,2) values,
% similar to factorizeS except that
% C(k,1) can be a cell array of names (instead of one name),
% in which case the fields will change values together.
%
% fields: fields to include or exclude
% to_fix
% : if true, fix the specified fields (do not factorize them).
% : if false (default), factorize only the specified fields.
%   When fixing a field, the field is fixed to value{1}, ignoring the rest.
%
% EXAMPLES:
% %% Factorize a and the b-c pair.
% [Ss, n] = factorizeC({
%     'a', {1, 2}
%     {'b', 'c'}, {{10, 100}, {20, 200}}
%     });
% for ii = 1:n
%     disp(Ss(ii));
% end
%
% %% RESULTS:
%     a: 1
%     b: 10
%     c: 100
% 
%     a: 1
%     b: 20
%     c: 200
% 
%     a: 2
%     b: 10
%     c: 100
% 
%     a: 2
%     b: 20
%     c: 200
%
% %% Factorize a only
% [Ss, n] = factorizeC({
%     'a', {1, 2}
%     {'b', 'c'}, {{10, 100}, {20, 200}}
%     }, 'a');
% for ii = 1:n
%     disp(Ss(ii));
% end
% 
% %% RESULTS:
%     a: 1
%     b: 10
%     c: 100
% 
%     a: 2
%     b: 10
%     c: 100
%
% %% Factorize all but a
% [Ss, n] = factorizeC({
%     'a', {1, 2}
%     {'b', 'c'}, {{10, 100}, {20, 200}}
%     }, 'a', true);
% for ii = 1:n
%     disp(Ss(ii));
% end
%
% %% RESULTS:
%     b: 10
%     c: 100
%     a: 1
% 
%     b: 20
%     c: 200
%     a: 1
% 
% See also: factorizeS, factorize

% 2016 (c) Yul Kang. hk2699 at columbia dot edu.

if isempty(C)
    Ss = struct;
    n = 1;
    return;
end

C = S2C2(C);
m = size(C,1);

% Handle empty cases
if isempty(C)
    Ss = struct;
    n = 1;
    return;
end

if nargin < 2
    fields = C(:,1);
elseif ~iscell(fields)
    fields = {fields};
end 
if nargin < 3
    to_fix = false; 
end
   
if to_fix
    incl = true(m, 1);
else
    incl = false(m, 1);
end

nf = numel(fields);
for ii = 1:m
    if ~iscell(C{ii,2})
        C{ii,2} = C(ii,2);
    end
    
    for jj = 1:nf
        if isequal(C{ii,1}, fields{jj})
            incl(ii) = ~to_fix;
            break;
        end
    end
end
C_fixed = C(~incl,:);
C = C(incl, :);
nf_fac = size(C, 1);
nf_fix = size(C_fixed, 1);

[fac, n] = factorize(C(:,2));

for ii = n:-1:1
    for jj = 1:nf_fac
        C1 = C{jj,1};
        C2 = fac{ii,jj};
    
        if ischar(C1)
            Ss(ii).(C1) = C2;
        else
            assert(iscell(C1));
            assert(all(cellfun(@ischar, C1)));
            assert(numel(C1) == numel(C2));

            for kk = 1:numel(C1)
                Ss(ii).(C1{kk}) = C2{kk};
            end
        end
    end
end

for ii = n:-1:1
    for jj = 1:nf_fix
        C1 = C_fixed{jj,1};
        C2 = C_fixed{jj,2}{1};
        
        if ischar(C1)
            Ss(ii).(C1) = C2;
        else
            assert(iscell(C1));
            assert(all(cellfun(@ischar, C1)));
            assert(numel(C1) == numel(C2));

            for kk = 1:numel(C1)
                Ss(ii).(C1{kk}) = C2{kk};
            end
        end
    end
end

return;

%% Examples
%% Factorize a and the b-c pair.
[Ss, n] = factorizeC({
    'a', {1, 2}
    {'b', 'c'}, {{10, 100}, {20, 200}}
    });
for ii = 1:n
    disp(Ss(ii));
end

%% Factorize a only
[Ss, n] = factorizeC({
    'a', {1, 2}
    {'b', 'c'}, {{10, 100}, {20, 200}}
    }, 'a');
for ii = 1:n
    disp(Ss(ii));
end

%% Factorize all but a
[Ss, n] = factorizeC({
    'a', {1, 2}
    {'b', 'c'}, {{10, 100}, {20, 200}}
    }, 'a', true);
for ii = 1:n
    disp(Ss(ii));
end
