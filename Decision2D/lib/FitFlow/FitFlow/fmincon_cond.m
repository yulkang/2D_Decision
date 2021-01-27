function C = fmincon_cond(nam, constr)
% C = fmincon_cond(nam, conds)
%
% nam   : A cell array of parameter names.
% conds : A cell array of conditions.
% C : A cell array containing {A, b, Aeq, beq, nonlcon}.
%
% EXAMPLE:
% >> conds = {
%     {'A',   {'th1', 'th3'}, [1 -1], 2}            % th1 - th2 <= 2
%     {'A',   {'th2', 'th4'}, [1 -1], 2}            % th1 - th2 <= 2
%     {'Aeq', {'th1', 'th3'}, [1  1], 3}            % th1 + th3 == 3
%     {'Aeq', {'th2', 'th4'}, [1  1], 3}            % th1 + th3 == 3
%     {'c'    {'th1', 'th3'}, @(v) prod(v) - 5}     % th1 * th2 <= 5
%     {'c'    {'th2', 'th4'}, @(v) prod(v) - 5}     % th1 * th2 <= 5
%     {'ceq', {'th1', 'th3'}, @(v) prod(v) - 7}     % th1 * th3 == 7
%     {'ceq', {'th2', 'th4'}, @(v) prod(v) - 7}     % th1 * th3 == 7
%    };
%
% >> C = fmincon_cond({'th1', 'th2', 'th3', 'th4'}, conds);
%
% >> C{:}
% ans =
%      1     0    -1     0
%      0     1     0    -1
% ans =
%      2     2
% ans =
%      1     0     1     0
%      0     1     0     1
% ans =
%      3     3
% ans = 
%     @fmincon_cond/f_nonlcon
%
% >> [c, ceq] = C{5}(1:4)
% c =       3   % 1*2-5 = -3 <= 0, 2*4-5 = 3 > 0, so 3 is the first positive answer.
% ceq =    -4   % 1*2-7 = -4 ~= 0, so -4 is the first nonzero answer.
%
% When there are multiple nonlcons for c, 
% the final c is the first positive c, or the last c.
%
% When there are multiple nonlcons for ceq,
% ceq is zero when all are zero, and the last value otherwise.
%
% See also: fmincon
%
% 2015 (c) Yul Kang. yul dot kang dot on at gmail dot com.

n = length(constr);
p = length(nam);

A       = [];
b       = [];
Aeq     = [];
beq     = [];
cs      = {};
ceqs    = {};
nonlcon = [];

nA      = 0;
nAeq    = 0;
nc      = 0;
nceq    = 0;

for ii = 1:n
    ix = strcmpfinds(constr{ii}{2}, nam);
    
    switch constr{ii}{1}
        case 'A'
            if nA == 0, A = zeros(1, p); end
            
            nA           = nA + 1;
            A(nA,ix)     = constr{ii}{3}; %#ok<*AGROW>
            b(nA)        = constr{ii}{4};
            
        case 'Aeq'
            if nAeq == 0, Aeq = zeros(1, p); end
            
            nAeq         = nAeq + 1;
            Aeq(nAeq,ix) = constr{ii}{3};
            beq(nAeq)    = constr{ii}{4};
            
        case 'c'
            nc           = nc + 1;
            cs{nc}       = @(th) constr{ii}{3}(th(ix));
            
        case 'ceq'
            nceq         = nceq + 1;
            ceqs{nceq}   = @(th) constr{ii}{3}(th(ix));
    end
end

if nc > 0 || nceq > 0
    nonlcon = @f_nonlcon;
end

C = {A, b, Aeq, beq, nonlcon};

% Nested functions
    function [c, ceq] = f_nonlcon(th)
        c = zeros(1, nc);
        for ic = 1:nc
            c(ic)  = cs{ic}(th);
%             if c > 0, break; end
        end
        
        ceq = zeros(1, nceq);
        for iceq = 1:nceq
            ceq(iceq)  = ceqs{iceq}(th);
%             if ceq ~= 0, break; end
        end
    end
end