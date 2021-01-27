function colors = winter2(n)
% colors = WINTER2(n) 
% Similar to MATLAB's WINTER but a bit stronger.

% Yul Kang, hk2699 at caa dot columbia dot edu.

if nargin < 1
    n = 256;
end

colors = linspaceN([0, 0.4, 1], [0, 0.8, 0.25], n);