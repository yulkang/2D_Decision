classdef FminconReduce < FitReduce
%
% 2015 (c) Yul Kang. yul dot kang dot on at gmail dot com.
methods (Static)
function varargout = fmincon(varargin)
    [fun2, x02, A2, b2, Aeq2, beq2, lb2, ub2, nonlcon2, opt, to_fix] = ...
        FminconReduce.replace_args(varargin{:});
    [varargout{1:nargout}] = fmincon( ...
        fun2, x02, A2, b2, Aeq2, beq2, lb2, ub2, nonlcon2, opt);
    
    x0 = varargin{2};
    [varargout{1:nargout}] = ...
        FminconReduce.replace_outputs(x0, to_fix, varargout{:});
end
function varargout = replace_outputs(x0, to_fix, varargin)
    varargout = varargin;
    if nargout >= 1
        % x
        varargout{1} = FminconReduce.fill_vec(x0, to_fix, varargout{1});
    end
    if nargout >= 6
        % gradient
        if isempty(varargout{6})
            varargout{6} = nan(1, length(x0));
        else
            varargout{6} = FminconReduce.fill_vec(zeros(size(to_fix)), to_fix, varargout{6});
        end
    end
    if nargout >= 7
        % Hessian
        if isempty(varargout{7})
            varargout{7} = nan(length(x0));
        else
            varargout{7} = FminconReduce.fill_mat(varargout{7}, to_fix, inf);
        end
    end
end
function [fun2, x02, A2, b2, Aeq2, beq2, lb2, ub2, nonlcon2, opt, to_fix] = ...
    replace_args(fun, x0, A, b, Aeq, beq, lb, ub, nonlcon, opt, to_fix)
    % [fun2, x02, A2, b2, Aeq2, beq2, lb2, ub2, nonlcon2, to_fix] = ...
    %     repace_args(fun, x0, A, b, Aeq, beq, lb, ub, nonlcon, to_fix)

    if nargin < 3, A = []; end
    if nargin < 4, b = []; end
    if nargin < 5, Aeq = []; end
    if nargin < 6, beq = []; end
    if nargin < 7, lb = -inf(size(x0)); end
    if nargin < 8, ub = inf(size(x0)); end
    if nargin < 9, nonlcon = []; end
    if nargin < 10, opt = []; end
    if nargin < 11 || isempty(to_fix)
        to_fix = isequal_mat_nan(lb, ub); 
    else
        lb(to_fix) = x0;
        ub(to_fix) = x0;
    end
    
    F = FminconReduce; % shortcut

    % Not using shortcut in function handle, so that it works after
    % converting to string and back.
    fun2 = @(v) FminconReduce.postprocess_output(wrap( ...
            @() fun(FminconReduce.fill_vec(x0, to_fix, v)), ...
        1:nargout), to_fix);
    x02  = x0(~to_fix);
    
    % Empty defaults
    A2 = [];
    b2 = [];
    Aeq2 = [];
    beq2 = [];
    lb2 = [];
    ub2 = [];
    nonlcon2 = [];
    
    % Work on each output
    if ~isempty(A)
        [A2, b2] = F.reduce_constr(A, b, x0, to_fix, @(Ax,b) Ax <= b);
    end
    if ~isempty(Aeq)
        [Aeq2, beq2] = F.reduce_constr(Aeq, beq, x0, to_fix, @(Ax,b) Ax == b);
    end
    if ~isempty(lb)
        lb2 = lb(~to_fix);
    end
    if ~isempty(ub)
        ub2 = ub(~to_fix);
    end
    if ~isempty(nonlcon)
        nonlcon2 = @(v) nonlcon(FminconReduce.fill_vec(x0, to_fix, v));
    end
end
function varargout = postprocess_output(outs, to_fix)
    if nargout >= 1 % cost
        varargout{1} = outs{1};
    end
    if nargout >= 2 % gradient
        varargout{2} = outs{2}(~to_fix);
    end
    if nargout >= 3 % hessian
        varargout{3} = outs{3}(~to_fix, ~to_fix);
    end
end
function v = fill_vec(x0, to_fix, x_vary)
    % v = fill_vec(x0, fixed, to_vary)
    siz = size(x0);
    v = zeros(siz);
    v(to_fix) = x0(to_fix);
    v(~to_fix) = x_vary;
end
function v2 = fill_mat(v, to_fix, v_diag)
    % v2 = fill_mat(v, to_fix, v_diag)
    % Compose a full matrix v2 with varying parameters' matrix v and fixed parameters' matrix v_diag.
    v2(~to_fix, ~to_fix) = v;
    v2(to_fix, to_fix) = diag(v_diag + zeros(1, nnz(to_fix)));
end
function [A2, b2] = reduce_constr(A, b, x0, to_fix, cond)
    % [A2, b2] = reduce_constr(A, b, x0, to_fix, cond)
    A2 = A(:,~to_fix);
    
    nonzero_A = any(A, 1);
    nonzero_fixed = to_fix(:) & nonzero_A(:);
    contrib_A = bsxfun(@times, A(:,nonzero_fixed), hVec(x0(nonzero_fixed)));
    
    b2 = b(:) - sum(contrib_A, 2);
    
    null_rows = ~any(A2, 2);
    if any(~cond(0, b2(null_rows))) % Since null rows of A2 always gives zero.
        error('x0 violates %s', func2str(cond));
    end
    
    A2 = A2(~null_rows, :);
    b2 = b2(~null_rows);
end
end
end  