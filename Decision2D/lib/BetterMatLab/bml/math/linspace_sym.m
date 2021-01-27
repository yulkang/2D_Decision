function v = linspace_sym(m, w)
% Regularly spaced vector, symmetric around zero.
%
% v = linspace_sym(m, w)
%
% EXAMPLE:
% >> linspace_sym(1, 0.3)
% ans =
%     0.9000    0.6000    0.3000         0    0.3000    0.6000    0.9000

v = [-fliplr(0:w:m), w:w:m];