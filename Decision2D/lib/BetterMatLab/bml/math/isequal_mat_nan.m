function c = isequal_mat_nan(a, b)
% Returns a matrix like ==, but treating NaNs to be equal.
%
% c = isequal_mat_nan(a, b)
%
% c = (a == b) | (isnan(a) & isnan(b));
c = (a == b) | (isnan(a) & isnan(b));