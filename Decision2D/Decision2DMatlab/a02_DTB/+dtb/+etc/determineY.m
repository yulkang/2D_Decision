function [y, y0, dy, ny] = determineY(k, coh, dt, ub, lb, ny, sig, marginFactor) % , granularity)
% [y, y0, dy, ny] = determineY(k, coh, dt, ub, lb, ny, sig, marginFactor) % , granularity)
%
% Determines y for spectral_dtb type functions.
%
% Output:
% y  : y vector. Always includes y==0.
% y0 : 1 at y==0.
%
% Required arguments:
%
% k  : multiplies coh
% coh: vector of coherences
% dt : size of the time step
% ub : upper bound. Can be a vector.
% lb : lower bound. Can be a vector.
%
% Optional arguments:
%
% ny : minimum length of y, say, 2^12. 
% sig: standard deviation.
% marginFactor: minimum margin as a factor of sig.
% granularity: minimum granularity that divides sig. % unused.

% Yul Kang (c) 2013, hk2699 at columbia.

if ~exist('ny' , 'var') || isempty(ny), ny  = 2^12; end
if ~exist('sig', 'var') || isempty(sig), sig = 1; end
if ~exist('lb', 'var')  || isempty(lb), lb = -ub; end
if ~exist('marginFactor', 'var') || isempty(marginFactor), marginFactor = 10; end
% if ~exist('granularity', 'var')  || isempty(granularity), granularity = 20; end

eCoh = k*coh*dt;
eSig = sig*sqrt(dt);

y_min = min(min(lb), min(eCoh)) - marginFactor*eSig;
y_max = max(max(ub), max(eCoh)) + marginFactor*eSig;

[y, y0, dy] = dtb.etc.determine_y_from_range(y_min, y_max, ny);