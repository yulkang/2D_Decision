function tf = ix2tf(siz, ix)
% tf = ix2tf(siz, ix)
%
% See also: data, PsyLib
%
% 2013 (c) Yul Kang. See help PsyLib for the license.

if isscalar(siz), siz = [1, siz]; end
tf = false(siz);
tf(ix) = true;
