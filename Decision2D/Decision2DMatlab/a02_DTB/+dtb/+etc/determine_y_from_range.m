function [y, p0, dy] = determine_y_from_range(y_min, y_max, ny, y0)
% [y, p0, dy] = determine_y_from_range(y_min, y_max, ny, y0)

% Yul Kang (c) 2013, hk2699 at columbia.

dy   = min((y_max - y_min) / ny); % , eSig/granularity);

n_neg = ceil(ny * -y_min / (y_max - y_min)); % n_neg should be positive, y_min is negative.
n_pos = ny - n_neg - 1; % n_neg + n_pos + 1 = ny. % +1 to include 0.

y     = vVec((-n_neg : n_pos) * dy);

if nargin < 4, y0 = 0; end
if nargout >= 2
    p0 = delta_fun(y0, y);
end
end