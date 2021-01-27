function strt = packStruct(varargin)
% Packs variables into a struct, using the variables' name as the fields' name.
%
% res = packStruct(var1, var2, ...)
%
% res.var1 == var1, and so on.
%
% See also: data, PsyLib
%
% 2013 (c) Yul Kang. See help PsyLib for the license.

strt = struct;

for ii = 1:length(varargin)
    strt.(inputname(ii)) = varargin{ii};
end