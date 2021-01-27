function [tf, a, b, sub, v] = assert_isequal_within(a, b, tol, varargin)
% [tf, a, b, sub, v] = assert_isequal_within(a, b, tol=1e-6, ...)
%
% a dictates the size used in the report when there is discrepancy.
%
% OPTION
% ------
% 'relative_tol', true

if nargin < 3, tol = 1e-6; end

S = varargin2S(varargin, {
    'relative_tol', true
    });

if S.relative_tol
    tol = max(max(abs(a(:))), max(abs(b(:)))) * tol;
end

dif = a - b;

try
    assert(max(abs(dif(:))) <= tol);
catch err
    [v, ind] = max(abs(dif(:)));
    
    [a, b] = rep2match({a, b});
    
    sub = bml.matrix.ind2sub_mat(size(a), ind);
    fprintf('Max |a - b| at (');
    fprintf('%d,', sub);
    fprintf(') : ');
    fprintf('%1.3g\n', v);
    
    inds = num2cell(ind);
    fprintf('a = %1.3g\n', a(inds{:}));
    fprintf('b = %1.3g\n', b(inds{:}));
    
    if nargout > 0
        tf = false;
        return;
    else
        rethrow(err);
    end
end
if nargout > 0
    tf = true;
    v = [];
    sub = [];
end