function p = invLogit(x)
% p = invLogit(x)
%   = exp(x)./(1+exp(x));

% 2021 Yul Kang. hk2699 at caa dot columbia dot edu.

p = exp(x)./(1+exp(x));
p_nan = isnan(p);
if any(p_nan)
    x_pos = x > 0;
    x_neg = x < 0;
    
    p(x_pos & p_nan) = 1;
    p(x_neg & p_nan) = 0;
    
    % This leaves p(isnan(x)) == nan, which is correct.
end
end