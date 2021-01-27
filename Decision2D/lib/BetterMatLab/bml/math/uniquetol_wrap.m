function [c, ia, ic] = uniquetol_wrap(a, tol)
% Deprecated, now that Shadlen lab cluster has MATLAB 2015a.
%
% [c, ia, ic] = uniquetol_wrap(a, tol)
% Mimics uniquetol for compatibility.
% Results are different from uniquetol when there are many similar values, 
% Use only to tolerate small numerical errors with a generous tolerance level!
%
% [c, ia, ic] = uniquetol_wrap(a, tol)
%
% 2015 (c) Yul Kang. hk2699 at cumc dot columbia dot edu.

if nargin < 2
    if isa(a, 'double')
        tol = 1e-12;
    elseif isa(a, 'single')
        tol = 1e-6;
    % otherwise, must issue an error.
    end
end

if exist('uniquetol', 'builtin')
    [c, ia, ic] = uniquetol(a, tol);
else
    tol = tol * max(abs(a)) * 2;
    [~, ia, ic] = unique(round(a/tol));
    c = a(ia);
end