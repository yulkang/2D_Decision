function [Ss, n] = factorizeS(S, fields, to_fix)
% [Ss, n] = factorizeS(S, fields=(all), to_fix=false)
%
% Alias to factorizeC, a new, more general version.
%
% See also: factorizeC

    if nargin < 2, fields = fieldnames(S); end
    if nargin < 3, to_fix = false; end
    [Ss, n] = factorizeC(S, fields, to_fix);

%     % [Ss, n] = factorizeS(S, fields=(all), to_exclude=false)
%     %
%     % Factorize fields of a scalar sturct, and return a struct array.
%     % Use cell arrays for each field to avoid confusion.
%     %
%     % Unlike the previous version,
%     % all non-cell arrays will be converted to a scalar cell array,
%     % including an empty entry ('' or []) or a string ('abc').
%     %
%     % Excluded fields will not be factorized, but will be
%     % included as a field for each element in the returned struct array.
%     %
%     % 2014-2015 (c) Yul Kang. yul dot kang dot on at gmail dot com.
%     
%     assert(isstruct(S) && isscalar(S));
%     
%     if nargin < 2, fields = fieldnames(S); end
%     if nargin < 3, to_exclude = false; end
%     
%     C = struct2cell(S);
%     
%     fields0 = fieldnames(S);
%     incl = strcmps(fields, fields0);
%     if to_exclude
%         incl = ~incl; 
%     end
%     if any(~incl)
%         C0 = cell(size(C));
%         C0(~incl) = C(~incl);
%     end
%     fields = fields0(incl);
%     C = C(incl);
%     
%     for ii = 1:numel(C)
%         if ~iscell(C{ii}) % Behavior different from previous version
%             C{ii} = {C{ii}}; %#ok<CCAT1>
%         end
%     end
%     
%     [Cs, n] = factorize(C);
%     Ss = cell2struct(Cs, fields, 2);
%     
%     if any(~incl)
%         for ii = find(~incl(:)')
%             for jj = 1:numel(Ss)
%                 Ss(jj).(fields0{ii}) = C0{ii};
%             end
%         end
%     end
end
