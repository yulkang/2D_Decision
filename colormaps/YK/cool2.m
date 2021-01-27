function colors = cool2(n)
% colors = COOL2(n) 
% Similar to MATLAB's COOL but a bit stronger.

% Yul Kang, hk2699 at caa dot columbia dot edu.

if nargin < 1
    n = 256;
end

colors = linspaceN([0.4, 0, 1], [1, 0, 0], n);